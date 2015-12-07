require 'torch'
local util = {}
function util.one_hot(dim,ind)
    local vec = torch.zeros(dim)
    vec[ind] = 1
    return vec
end
function util.get_ind(vec)
    local numbers = torch.range(1,(#vec)[1])
    return numbers[vec:byte()][1]
end
return util
