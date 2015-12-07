local ClockLin, parent = torch.class('nn.ClockLin', 'nn.Module')

function ClockLin:__init(inputSize, outputSize,inputType)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   if inputType == 'input' then
        self.maskdW = true
   end

   self:reset()
end

function ClockLin:reset(stdv)
    --self.weight:normal(0,.1)
--
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
--]]
   return self
end

function ClockLin:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, torch.cmul(self.mask,self.weight), input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
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

function ClockLin:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
            --don't pass back unless you passed forward
            self.gradInput:addmv(0, 1, torch.cmul(self.mask,self.weight):t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end
function ClockLin:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
        if self.maskdW then
            --the state didn't contribute, so dont change its weights
            self.gradWeight:addr(scale, torch.cmul(self.actMask,gradOutput), input)
        else
            --accum gradient for clocks, since we know input ultimately responsible
            self.gradWeight:addr(scale, gradOutput, input)
        end
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
end

-- we do not need to accumulate parameters when sharing
ClockLin.sharedAccUpdateGradParameters = ClockLin.accUpdateGradParameters


function ClockLin:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
