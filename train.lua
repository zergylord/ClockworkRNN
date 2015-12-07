--ProFi = require 'ProFi'
--ProFi:start()
require 'Clockwork'
require 'nngraph'
require 'optim'
require 'gnuplot'
util = require 'util.model_utils'
target = torch.load('music.t7')
target:add(-.5):mul(2)
in_pool = nn.Identity()()
rec_pool = nn.Identity()()
num_hid = 49
--
cw = nn.Clockwork(1,num_hid,7)
layer = cw{in_pool,rec_pool}:annotate{name='clock'}
--[[
inlin = nn.Linear(1,num_hid)(in_pool)
reclin = nn.Linear(num_hid,num_hid)(rec_pool)
layer = nn.Tanh()(nn.CAddTable(){inlin,reclin})
--]]
out_pool= nn.Linear(num_hid,1)(layer)
network = nn.gModule({in_pool,rec_pool},{out_pool,layer})
parameters, gradients = network:getParameters()
network:zeroGradParameters()
timer = torch.Timer()
max_steps = 100
for i,node in ipairs(network.forwardnodes) do
    if node.data.annotations.name == 'clock' then
        clock_node_ind = i
        break
    end
end
local net_clones = util.clone_many_times(network,max_steps)

if clock_node_ind then
    print('setting clocks')
    for i=1,max_steps do
        net_clones[i].forwardnodes[clock_node_ind].data.module:sett(i-1)
    end
end

local mse_crit = nn.MSECriterion()
local opfunc = function(x)
    if x ~= parameters then
        parameters:copy(x)
    end
    prev_grad = torch.zeros(num_hid)
    network:zeroGradParameters()
    data = {}
    output = torch.zeros(max_steps)
    rec = {}
    rec[0] = torch.zeros(num_hid)
    for t = 1,max_steps do
        --cw:sett(t-1)
        data[t] = {torch.zeros(1),rec[t-1]}
        output[t],rec[t] = unpack(net_clones[t]:forward(data[t]))
        --[[
        print(output)
        gnuplot.plot(output)
        gnuplot.plotflush()
        os.execute('sleep 1')
        --]]
    end
    loss = 0
    for t = max_steps,1,-1 do
        loss = loss + mse_crit:forward(torch.Tensor{output[t]},torch.Tensor{target[t]})
        local grad = mse_crit:backward(torch.Tensor{output[t]},torch.Tensor{target[t]})
        if prev_grad:norm() > 1e7 or prev_grad:norm() ~= prev_grad:norm() then
           -- print(prev_grad)
            print('crash')
            os.exit()
        end
        _,prev_grad = unpack(net_clones[t]:backward(data[t],{grad,prev_grad}))
    end
    return loss,gradients
end
config = {
    learningRate = 3e-4, momentum = .95, nesterov = true, dampening = 0
}
local cumloss = 0
for i = 1,1e5 do
    x, batchloss = optim.sgd(opfunc, parameters, config)
    --[[
    if i == 10 then
        os.exit()
    end
    --]]
    cumloss = cumloss + batchloss[1]
    --print(gradients)
    --print(net_clones[1].forwardnodes[clock_node_ind].data.module.net:parameters()[3])
    if i % 1e3 == 0 then
        print(i,cumloss,gradients:norm(),timer:time().real)
        timer:reset()
        gnuplot.plot({target},{output})
        cumloss = 0
        collectgarbage()
    end
end
--ProFi:stop()
--ProFi:writeReport('train_report.txt')
