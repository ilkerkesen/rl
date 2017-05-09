using Knet
using ReinforcementLearning
using ArgParse
using JLD

function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        # load/save files
        ("--lr"; default=0.001; arg_type=Float64)
        ("--batchsize"; default=128; arg_type=Int64)
        ("--units"; default=200; arg_type=Int64)
        ("--gamma"; default=0.95; arg_type=Float64)
        ("--epsinit"; default=0.9; arg_type=Float64)
        ("--epsfinal"; default=0.005; arg_type=Float64)
        ("--epsdecay"; default=200; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}":"Array{Float32}"))
        ("--nepisodes"; default=10; arg_type=Int64)
        # ("--dist";arg_type=String;default="randn";help="[randn|xavier]")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()
end

function initweights(atype)
    w = Array(Any,8)
    w[1] = xavier(5,5,3,16)
    w[2] = zeros(1,1,16,1)
    w[3] = xavier(5,5,16,32)
    w[4] = zeros(1,1,32,1)
    w[5] = xavier(5,5,32,32)
    w[6] = zeros(1,1,32,1)
    w[7] = xavier(2,448)
    w[8] = zeros(2,1)
    return map(wi->convert(atype,wi), w)
end

function predict(w,x0)
    cbr(x,i) = relu(conv4(w[2i-1], x; padding=0, stride=2) .+ w[2i])

    x1 = cbr(x0,1)
    x2 = cbr(x1,2)
    x3 = cbr(x2,3)
    x4 = w[end-1] * mat(x4) .+ w[end]
end

type ReplayMemory
    capacity
    memory

    function ReplayMemory(capacity)
        memory = Transition[]
        new(capacity, memory)
    end
end

type Transition
    state
    action
    next_state
    reward
end

function push!(obj::ReplayMemory, t::Transition)
    push!(obj.memory,t)
    length(obj.memory) > obj.capacity && shift!(obj.memory)
end

function length(obj::ReplayMemory)
    return length(obj.memory)
end

function sample(obj::ReplayMemory, batchsize)
    indices = randperm(length(obj))[1:batchsize]
    return obj.memory[indices]
end
