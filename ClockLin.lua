local ClockLin, parent = torch.class('nn.ClockLin', 'nn.Module')

function ClockLin:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize+outputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize+outputSize)
   self.gradBias = torch.Tensor(outputSize)

   self:reset()
end

function ClockLin:updateOutput(input)
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, torch.cmul(self.mask,self.weight), input)
   return self.output
end

function ClockLin:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      self.gradInput:addmv(0, 1, 
          torch.cmul(self.mask,self.weight):t(),
          gradOutput)
      return self.gradInput
   else
    print('what?')
    end
end
function ClockLin:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight:addr(scale, gradOutput, input)
   self.gradBias:add(scale, gradOutput)
end

-- we do not need to accumulate parameters when sharing
ClockLin.sharedAccUpdateGradParameters = ClockLin.accUpdateGradParameters


function ClockLin:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
