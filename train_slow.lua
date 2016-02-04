timer = torch.Timer()
util = require 'util.model_utils'
require 'Clockwork_slow'
require 'optim'
require 'gnuplot'
input = nn.Identity()()
rec = nn.Identity()()
num_clocks = 7
num_tot = 7
num_out = 1
num_in = 1
--hid = nn.Tanh()(nn.Linear(num_tot+num_in,num_tot)(nn.JoinTable(1){input,rec}))
--
a = nn.Clockwork(num_in,num_tot,num_clocks)
hid = a{input,rec}:annotate{name='clock'}
--]]
local layer = nn.Linear(num_tot,num_out)
layer.weight:normal(0,.1)
out = layer(hid)
net = nn.gModule({input,rec},{out,hid})
w,dw = net:getParameters()
net:zeroGradParameters()
max_steps = 100
net_clones = util.clone_many_times(net,max_steps)
for i,node in ipairs(net.forwardnodes) do
    if node.data.annotations.name == 'clock' then
        clock_node_ind = i
        break
    end
end
if clock_node_ind then
    print('setting clocks')
    for t=1,max_steps do
        net_clones[t].forwardnodes[clock_node_ind].data.module:setTime(t-1)
    end
end
--local target = torch.linspace(0,1,max_steps)
local target = torch.linspace(-1,1,50):cat(torch.linspace(-1,1,50))
local mse_crit = nn.MSECriterion()
local cumtime = 0
opfunc = function (x)
    if x ~= w then
        w:copy(x)
    end
    net:zeroGradParameters()
    data = {}
    y = torch.zeros(max_steps)
    rec_state = torch.zeros(num_tot)
    --timer:reset()
    for t = 1,max_steps do
        --from environment
        data[t] = {torch.zeros(1),rec_state:clone()}
        y[t],rec_state = unpack(net_clones[t]:forward(data[t]))
    end
    local loss = 0
    local prev_grad = torch.zeros(num_tot)
    for t = max_steps,1,-1 do
        loss = loss + mse_crit:forward(torch.Tensor{y[t]},torch.Tensor{target[t]})
        local grad = mse_crit:backward(torch.Tensor{y[t]},torch.Tensor{target[t]})
        _,prev_grad = unpack(net_clones[t]:backward(data[t],{grad,prev_grad}))
    end
    --cumtime = cumtime + timer:time().real
    return loss,dw
end
config = {
    learningRate = 3e-4, momentum = .95, nesterov = true, dampening = 0
}
local cumloss = 0
for i = 1,1e5 do
    x, batchloss = optim.sgd(opfunc, w, config)
    cumloss = cumloss + batchloss[1]
    if i%1e3 == 0 then
        print(i,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        gnuplot.plot({target},{y})
        cumloss = 0
        cumtime = 0
        collectgarbage()
    end
end


