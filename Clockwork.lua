require 'nn'
require 'ClockLin'
local Clockwork, parent = torch.class('nn.Clockwork', 'nn.Module')

function Clockwork:__init(inputSize, outputSize, numClocks)
   parent.__init(self)
   self.numClocks = numClocks
   self.net = nn.Sequential()
   local par = nn.ParallelTable()
   self.linLayer = nn.ClockLin(inputSize,outputSize,'input')
   self.recLayer = nn.ClockLin(outputSize,outputSize,'rec')
   par:add(self.linLayer)
   par:add(self.recLayer)
   self.net:add(par)
   self.net:add(nn.CAddTable())
   self.net:add(nn.Tanh())

   self.linMask = torch.ones(outputSize,inputSize) 
   self.recMask = torch.ones(outputSize,outputSize)

   self.gradInput = {}
   if outputSize % numClocks ~= 0 then
       error('inputs must be divisible by the number of clocks!')
   end
   self.clockSize = outputSize / numClocks
   self.mask = torch.zeros(outputSize,outputSize)
   for i=1,numClocks do
       self.mask[{{(i-1)*self.clockSize+1,i*self.clockSize},
                   {(i-1)*self.clockSize+1,numClocks*self.clockSize}}] = 1
   end
end

function Clockwork:parameters()
    return self.net:parameters()
end
--set time starting at 0
function Clockwork:sett(t)
    self.t = t
    local last
    for i=1,self.numClocks do
      if self.t % 2^(i-1) == 0 then
          last = i
      else
          break
      end
    end
    --print(t,last)
    self.onClocks = last
    self.recMask:copy(self.mask)
    self.linMask:zero():add(1)

    local start,stop = self.onClocks*self.clockSize+1,self.numClocks*self.clockSize 
    if start < stop then --noextra mask when all clocks on
        self.recMask[{{start,stop},{}}] = 0
        self.linMask[{{start,stop},{}}] = 0
    end
    self.linLayer.mask = self.linMask
    self.linLayer.actMask = self.linMask[{{},1}]
    self.recLayer.mask = self.recMask
end

function Clockwork:reset(stdv)
    self.net:reset()
    return  self
end


function Clockwork:updateOutput(input)
    self.output = self.net:forward(input)
    local actMask = self.linMask[{{},1}]:ne(1)
    self.output[actMask] = input[2][actMask]
   return self.output
end

function Clockwork:updateGradInput(input, gradOutput)
    self.gradInput = self.net:backward(input,gradOutput)
   return self.gradInput
end
