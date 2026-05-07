#include "close_chain_mapping.hpp"
#include "decouple_atom01.hpp"

std::shared_ptr<Decouple> Decouple::create(const std::string &type)
{
    if (type == "rp1")
    {
        return std::make_shared<DecoupleRP1>();
    }
    else if (type == "atom01")
    {
        return std::make_shared<DecoupleAtom01>();
    }
    else
    {
        throw std::runtime_error("Unknown close_chain type: " + type +
                                 ". Supported types: rp1, atom01");
    }
}
