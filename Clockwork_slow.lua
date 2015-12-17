require 'nngraph'
local Clockwork, parent = torch.class('nn.Clockwork', 'nn.Module')

function Clockwork:__init(inputSize, outputSize,numClocks)
   parent.__init(self)

    self.numClocks = numClocks
    self.num_tot = outputSize
    self.num_hid = outputSize/numClocks
    self.num_in = inputSize
    local num_params = 0
    self.w_ind = {1}
   for i=1,numClocks do
        local new_params = (self.num_hid*i + inputSize)*self.num_hid
        num_params = num_params + new_params 
        self.w_ind[i+1] = self.w_ind[i] + new_params
   end
   self.weight = torch.Tensor(num_params):normal(0,.1)
   self.bias = torch.Tensor(outputSize):normal(0,.1)
   self.gradWeight = torch.Tensor(num_params)
   self.gradBias = torch.Tensor(outputSize)

end
function Clockwork:reset(stdv)
    self.network:reset(stdv)
   return self
end

function Clockwork:setTime(t)
    local last = 1
    for i = 0,(self.numClocks-1) do
        if t % 2^i ~= 0 then
            break
        end
        last = i+1
    end
    self.last = last
    local input = nn.Identity()()
    local rec ={}
    local hid = {}
    local glue = {input}
    for i=1,self.numClocks do
        rec[i] = nn.Identity()()
        table.insert(glue,rec[i])
        if self.numClocks-i+1 <= last then
            local in_size = self.num_hid*i+self.num_in
            local params_used = in_size *self.num_hid
            local layer = nn.Linear(in_size,self.num_hid)
            layer.weight = self.weight[{{self.w_ind[i],self.w_ind[i+1]-1}}]
            layer.weight:resize(self.num_hid,in_size)
            layer.gradWeight = self.gradWeight[{{self.w_ind[i],self.w_ind[i+1]-1}}]
            layer.gradWeight:resize(self.num_hid,in_size)
            layer.bias = self.bias[{{self.num_hid*(i-1)+1,self.num_hid*i}}]
            layer.gradBias = self.gradBias[{{self.num_hid*(i-1)+1,self.num_hid*i}}]
            hid[i] = nn.Tanh()(layer(nn.JoinTable(1)(glue)))
        else
            hid[i] = rec[i]
        end 
    end
    self.network = nn.gModule(glue,hid)
end
function Clockwork:updateOutput(input)
    self.split_in = input[2]:split(self.num_hid)
    table.insert(self.split_in,1,input[1])
    local res = self.network:forward(self.split_in)
    self.output = torch.cat(res)
    return self.output
end
function Clockwork:updateGradInput(input, gradOutput)
    local res = self.network:backward(self.split_in,gradOutput:split(self.num_hid))
    self.gradInput =  {table.remove(res,1),torch.cat(res)}
    --self.gradInput = {torch.zeros(1),torch.zeros(49)}
    return self.gradInput
end

function Clockwork:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.num_in,self.num_tot)
end
