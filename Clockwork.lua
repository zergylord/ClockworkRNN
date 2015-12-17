require 'nn'
require 'ClockLin'
require 'gnuplot'
local Clockwork, parent = torch.class('nn.Clockwork', 'nn.Module')

function Clockwork:__init(inputSize, outputSize, numClocks)
   parent.__init(self)
    self.numClocks = numClocks
    self.num_tot = outputSize
   if outputSize % numClocks ~= 0 then
       error('inputs must be divisible by the number of clocks!')
   end
    self.num_hid = outputSize/numClocks
    self.num_in = inputSize
   self.weight = torch.Tensor(outputSize,inputSize+outputSize):normal(0,.1)
   self.bias = torch.Tensor(outputSize):normal(0,.1)
   self.gradWeight = torch.Tensor(outputSize,inputSize + outputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.numClocks = numClocks



   self.output = torch.zeros(outputSize)
   self.gradInput = {}
   self.mask = torch.zeros(self.weight:size())
   for i=1,numClocks do
       self.mask[{{(i-1)*self.num_hid+1,i*self.num_hid},
                   {(i-1)*self.num_hid+1,self.num_tot+self.num_in}}] = 1
   end
   --self:reset()
end

--set time starting at 0
function Clockwork:setTime(t)
    self.t = t
    local last
    for i=1,self.numClocks do
      if self.t % 2^(i-1) == 0 then
          last = i
      else
          break
      end
    end
    self.last = last
      

    local stop = last*self.num_hid 
    self.mask = self.mask[{{1,stop},{}}]
    self.act_mask = torch.zeros(self.num_tot):byte()
    self.act_mask[{{1,stop}}] =1 
    self.clock = nn.ClockLin(self.num_in+self.num_tot,stop)
    self.clock.mask = self.mask:double()
    self.clock.weight = self.weight[{{1,stop},{}}]
    self.clock.gradWeight = self.gradWeight[{{1,stop},{}}]
    self.clock.bias = self.bias[{{1,stop}}]
    self.clock.gradBias = self.gradBias[{{1,stop}}]
    self.net = nn.Sequential()
    self.net:add(self.clock)
    self.net:add(nn.Tanh())
end

function Clockwork:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end


function Clockwork:updateOutput(input)
    self.output = input[2]:clone()
    self.output[self.act_mask] = self.net:forward(torch.cat{input[2],input[1]})
    --[[
    gnuplot.bar(self.act_mask)
    gnuplot.plotflush()
    --]]
   return self.output
end

function Clockwork:updateGradInput(input, gradOutput)
    self.gradInput[2] = gradOutput:clone()
    local outputs = self.net:backward(torch.cat{input[2],input[1]},gradOutput[self.act_mask])
    self.gradInput[2][self.act_mask] = outputs[{{1,-self.num_in-1}}]
    self.gradInput[1] = outputs[{{-self.num_in,-1}}]
   return self.gradInput
end
