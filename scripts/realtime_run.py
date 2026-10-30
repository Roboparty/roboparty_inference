#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

import argparse
import errno
import fcntl
import os
from pathlib import Path
import pwd
import resource
import signal
import socket
import struct
import subprocess
import sys
import time

CGROUP = Path("/sys/fs/cgroup")


def user_context():
    env = dict(os.environ)
    uid = int(env.get("SUDO_UID", os.getuid()))
    gid = int(env.get("SUDO_GID", os.getgid()))
    user = pwd.getpwuid(uid)
    env.update(HOME=user.pw_dir, USER=user.pw_name, LOGNAME=user.pw_name)
    return env, uid, gid, user


def set_realtime_limits(uid, gid, user):
    resource.setrlimit(resource.RLIMIT_RTPRIO, (99, 99))
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    if uid:
        os.initgroups(user.pw_name, gid)
        os.setgid(gid)
        os.setuid(uid)


def run_affinity(command):
    env, uid, gid, user = user_context()

    def prepare_child():
        set_realtime_limits(uid, gid, user)

    print("Realtime isolation unavailable; using CPU affinity mode", file=sys.stderr, flush=True)
    return subprocess.call(command, env=env, preexec_fn=prepare_child if os.geteuid() == 0 else None)


def isolation_available(needs_cpuset=True):
    if os.geteuid() != 0:
        return False
    try:
        if needs_cpuset and "cpuset" not in (CGROUP / "cgroup.controllers").read_text().split():
            return False
        with os.fdopen(os.open("/dev/cpu_dma_latency", os.O_WRONLY | os.O_CLOEXEC),
                       "wb", buffering=0) as latency:
            latency.write(struct.pack("i", 0))
        return True
    except OSError:
        return False


def task_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def thread_ids(pid, expected):
    deadline = time.monotonic() + 5
    while True:
        found = {name: [] for name in expected}
        for task in Path(f"/proc/{pid}/task").iterdir():
            try:
                name = (task / "comm").read_text().strip()
                if name in found:
                    found[name].append(int(task.name))
            except FileNotFoundError:
                continue
        if all(len(found[name]) == count for name, count in expected.items()):
            return found
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Realtime thread count mismatch: expected {expected}, found {found}")
        time.sleep(0.01)


def realtime_run(command, can_count, canfd_count):
    if os.geteuid() != 0:
        return run_affinity(command)
    control_cpu, can_cpu, inference_cpu = (
        int(os.environ[name]) for name in ("ROBOT_CONTROL_CPU", "ROBOT_CAN_CPU", "ROBOT_INFERENCE_CPU"))
    cpus = [control_cpu, inference_cpu] + ([can_cpu] if can_count or canfd_count else [])
    if len(set(cpus)) != len(cpus):
        raise ValueError(f"Isolation CPUs must be distinct: {cpus}")
    isolated_path = Path("/sys/devices/system/cpu/isolated")
    isolated = set()
    if isolated_path.exists():
        for item in isolated_path.read_text().strip().split(","):
            if item:
                bounds = item.split("-")
                isolated.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    if not isolation_available(bool(set(cpus) - isolated)):
        return run_affinity(command)

    # Inherited housekeeping affinity is not the set of CPUs we may bind to.
    original = os.sched_getaffinity(0)
    try:
        for cpu in cpus:
            os.sched_setaffinity(0, {cpu})
    finally:
        os.sched_setaffinity(0, original)

    # The guardian owns restoration even if the launch supervisor is killed.
    with open("/run/roboparty-realtime.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        enabled = "cpuset" in (CGROUP / "cgroup.subtree_control").read_text().split()
        groups = [
            (CGROUP / f"roboparty-control-{os.getpid()}", control_cpu, ("control",)),
            (CGROUP / f"roboparty-inference-{os.getpid()}", inference_cpu, ("inference",)),
        ]
        if can_count or canfd_count:
            groups.append((CGROUP / f"roboparty-can-{os.getpid()}", can_cpu,
                           ("motor_can_rx", "motor_can_tx", "motor_canfd_rx", "motor_canfd_tx")))
        # Only partitions created here belong to the cleanup guardian.
        dynamic_groups = [group for group in groups if group[1] not in isolated]
        env, uid, gid, user = user_context()
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)

        def prepare_child():
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            set_realtime_limits(uid, gid, user)

        child = None
        try:
            child = subprocess.Popen(command, env=env,
                                     preexec_fn=prepare_child, start_new_session=True)
            identity = task_identity(child.pid)
            parent_guard, guardian_socket = socket.socketpair()
            guardian = os.fork()
        except BaseException:
            if child is not None:
                child.kill()
                child.wait()
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            raise
        if guardian == 0:
            parent_guard.close()
            os.setsid()
            for sig in signals:
                signal.signal(sig, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            try:
                guardian_socket.sendall(b"R")
                while guardian_socket.recv(1):
                    pass
            except OSError:
                pass
            guardian_socket.close()
            errors = []

            def restore(action):
                try:
                    action()
                except Exception as error:
                    errors.append(str(error))

            # A dead supervisor may leave its node alive; finish it before releasing the CPU.
            for sig, seconds in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
                if identity is None or task_identity(child.pid) != identity:
                    break
                restore(lambda: os.killpg(child.pid, sig))
                deadline = time.monotonic() + seconds
                while task_identity(child.pid) == identity and time.monotonic() < deadline:
                    time.sleep(0.05)
            for path, _, _ in reversed(dynamic_groups):
                if path.exists():
                    partition = path / "cpuset.cpus.partition"
                    if partition.exists():
                        restore(lambda partition=partition: partition.write_text("member"))
                    # Exiting threads can briefly keep their cgroup busy.
                    deadline = time.monotonic() + 2
                    while True:
                        try:
                            path.rmdir()
                            break
                        except OSError as error:
                            if error.errno != errno.EBUSY or time.monotonic() >= deadline:
                                errors.append(str(error))
                                break
                            time.sleep(0.01)
            def restore_controller():
                if enabled or "cpuset" not in (CGROUP / "cgroup.subtree_control").read_text().split():
                    return
                for path in CGROUP.iterdir():
                    if not path.is_dir():
                        continue
                    try:
                        explicit = ((path / "cpuset.cpus").read_text().strip() or
                                    (path / "cpuset.mems").read_text().strip() or
                                    "cpuset" in (path / "cgroup.subtree_control").read_text().split() or
                                    (path / "cpuset.cpus.partition").read_text().strip() != "member")
                    except FileNotFoundError:
                        continue
                    if explicit:
                        raise RuntimeError(f"Leaving cpuset enabled because another group uses it: {path}")
                (CGROUP / "cgroup.subtree_control").write_text("-cpuset")

            if dynamic_groups:
                restore(restore_controller)
            for error in errors:
                print(f"Realtime cleanup: {error}", file=sys.stderr, flush=True)
            os._exit(1 if errors else 0)

        guardian_socket.close()
        latency_fd = None
        guardian_ready = False

        def request_stop(sig, _frame):
            sys.exit(128 + sig)

        handlers = {sig: signal.signal(sig, request_stop) for sig in signals}
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            parent_guard.settimeout(5)
            if parent_guard.recv(1) != b"R":
                raise RuntimeError("Realtime cleanup guardian failed to start")
            guardian_ready = True
            expected = {"control": 1, "inference": 1,
                        "motor_can_rx": can_count, "motor_can_tx": can_count,
                        "motor_canfd_rx": canfd_count, "motor_canfd_tx": canfd_count,
                        }
            threads = thread_ids(child.pid, expected)
            try:
                if dynamic_groups:
                    (CGROUP / "cgroup.procs").write_text(str(child.pid))
                    if not enabled:
                        (CGROUP / "cgroup.subtree_control").write_text("+cpuset")
                for path, cpu, names in dynamic_groups:
                    path.mkdir()
                    (path / "cgroup.type").write_text("threaded")
                    (path / "cpuset.mems").write_text((CGROUP / "cpuset.mems.effective").read_text())
                    (path / "cpuset.cpus").write_text(str(cpu))
                    for name in names:
                        for member in threads[name]:
                            (path / "cgroup.threads").write_text(str(member))
                    (path / "cpuset.cpus.partition").write_text("isolated")
                    if (path / "cpuset.cpus.partition").read_text().strip() != "isolated":
                        raise RuntimeError(f"CPU{cpu} isolated partition is invalid")
                latency_fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY | os.O_CLOEXEC)
                if os.write(latency_fd, struct.pack("i", 0)) != struct.calcsize("i"):
                    raise RuntimeError("CPU latency request was not accepted")
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS, errno.EINVAL,
                                       errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOENT):
                    raise
                fallback_error = error
            else:
                print(f"Realtime: control CPU{control_cpu}, inference CPU{inference_cpu}, "
                      f"{2 * (can_count + canfd_count)} motor CAN threads on CPU{can_cpu}, "
                      "all isolated, latency=0 us",
                      flush=True)
                return child.wait()
        finally:
            for sig in signals:
                signal.signal(sig, signal.SIG_IGN)
            parent_guard.close()
            if not guardian_ready:
                child.kill()
            try:
                _, status = os.waitpid(guardian, 0)
                child.wait(timeout=2)
            finally:
                if latency_fd is not None:
                    os.close(latency_fd)
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            if status:
                raise RuntimeError("Realtime cleanup failed; see the cleanup diagnostics")
    print(f"Realtime isolation setup unavailable: {fallback_error}", file=sys.stderr, flush=True)
    return run_affinity(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-count", type=int, default=0)
    parser.add_argument("--canfd-count", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A node command is required")
    try:
        sys.exit(realtime_run(command, args.can_count, args.canfd_count))
    except (KeyError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(f"Realtime setup failed: {error}")
