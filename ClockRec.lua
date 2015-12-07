require 'nn'
local ClockRec, parent = torch.class('nn.ClockRec', 'nn.Module')

function ClockRec:__init(inputSize, outputSize, numClocks)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   self.numClocks = numClocks
    if inputSize % numClocks ~= 0 then
        error('inputs must be divisible by the number of clocks!')
    end
    self.clockSize = inputSize / numClocks
    self.mask = torch.zeros(outputSize,inputSize)
    for i=1,numClocks do
        self.mask[{{(i-1)*self.clockSize+1,i*self.clockSize},
                    {(i-1)*self.clockSize+1,numClocks*self.clockSize}}] = 1
    end
   self:reset()
end

--set time starting at 0
function ClockRec:sett(t)
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
    self.timeMask = self.mask:clone()
    local start,stop = self.onClocks*self.clockSize+1,self.numClocks*self.clockSize 
    if start < stop then --noextra mask when all clocks on
        self.timeMask[{{start,stop},{}}] = 0
    end
    --print(self.timeMask)
end

function ClockRec:reset(stdv)
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
   self:sett(0)
   return self
end

function ClockRec:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      local masked = self.weight:clone() 
      --print(self.onClocks)
      masked:cmul(self.timeMask)
      self.output:addmv(1, masked, input)
      --print('normal')
   elseif input:dim() == 2 then
      print('minibatch')
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
        print('weird')
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end

   return self.output
end
--only matters for passing back to input
function ClockRec:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, torch.cmul(self.weight,self.timeMask):t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function ClockRec:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
end

-- we do not need to accumulate parameters when sharing
ClockRec.sharedAccUpdateGradParameters = ClockRec.accUpdateGradParameters


function ClockRec:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
