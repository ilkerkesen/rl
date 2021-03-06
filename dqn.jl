using Knet
using ReinforcementLearning
using ArgParse
using JLD
import Base.push!
import Base.length

const SCREEN_WIDTH = 600
const BATCH_SIZE = 128
const GAMMA = 0.95
const EPS_INIT = 0.9
const EPS_FINAL = 0.005
const EPS_DECAY = 200
const NUM_EPISODES = 10
const LR = 0.001
const GCLIP = 5.0
const WINIT = 0.1
const CAPACITY = 10000
const PERIOD = 200

function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        ("--hidden"; nargs='*'; arg_type=Int64)
        ("--lr"; default=LR; arg_type=Float64)
        ("--gclip"; default=GCLIP; arg_type=Float64)
        ("--batchsize"; default=BATCH_SIZE; arg_type=Int64)
        ("--gamma"; default=GAMMA; arg_type=Float64)
        ("--epsinit"; default=EPS_INIT; arg_type=Float64)
        ("--epsfinal"; default=EPS_FINAL; arg_type=Float64)
        ("--epsdecay"; default=EPS_DECAY; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}":"Array{Float32}"))
        ("--winit"; default=WINIT; arg_type=Float64)
        ("--capacity"; default=CAPACITY; arg_type=Int64)
        ("--numepisodes"; default=NUM_EPISODES; arg_type=Int64)
        ("--monitor"; default=nothing)
        ("--seed"; default=-1; arg_type=Int64)
        ("--generate"; default=0; arg_type=Int64)
        ("--loadfile"; default=nothing)
        ("--savefile"; default=nothing)
        ("--period"; default=PERIOD; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()

    env = GymEnv("CartPole-v0")
    w = opts = nothing
    if o[:loadfile] == nothing
        w = initweights(o[:atype], o[:hidden], o[:winit])
    else
        w = load(o[:loadfile], "w")
        w = map(wi->convert(o[:atype], wi), w)
    end
    opts = map(wi->Adam(;gclip=o[:gclip]), w)

    memory = ReplayMemory(o[:capacity])
    if o[:monitor] != nothing
        monitor_start(env, o[:monitor])
    end

    steps_done = 0
    currerr = 0
    preverr = 0
    iter = 0
    for k = 1:o[:numepisodes]
        # reset environment
        s0 = getInitialState(env); state = s0

        while true
            s = convert(o[:atype], state.data)
            action = select_action(w,s,steps_done; o=o)
            steps_done += 1

            next_state, reward = transfer(env, state, env.actions[action])
            t = Transition(state, action, next_state, reward)
            push!(memory, t)
            state = next_state

            if length(memory) >= o[:batchsize]
                val = train!(w,memory,opts; o=o)

                preverr = currerr
                if iter < 100
                    currerr = (iter * currerr) + val
                    currerr = currerr / (iter+1)
                else
                    currerr = 0.01*val+0.99*currerr
                end

                iter += 1
                if iter % o[:period] == 0
                    println("($iter,$currerr)")
                end
            end

            if state.done
                break
            end
        end

        if o[:savefile] != nothing
            save(o[:savefile], "w", map(wi->Array(wi),w))
        end
    end

    for k = 1:o[:generate]
        done = false
        state = getInitialState(env)
        while !done
            s = convert(o[:atype], state.data)
            action = select_action(w,s,steps_done; o=o)
            steps_done += 1
            render_env(env)
            next_state, reward = transfer(env, state, env.actions[action])
            state = next_state
            done = next_state.done
        end
    end

    if o[:monitor] != nothing
        monitor_close(env)
    end
end

function initweights(atype, hidden, winit=0.1)
    w = []
    x = 4
    for y in [hidden..., 2]
        push!(w, winit*randn(y,x))
        push!(w, zeros(y,1))
        x = y
    end
    return map(wi->convert(atype,wi), w)
end

function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        if i < length(w)-1
            x = relu(x)
        end
    end
    return x
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
    batchsize = min(length(obj), batchsize)
    indices = randperm(length(obj))[1:batchsize]
    return obj.memory[indices]
end

function select_action(w, s, steps_done; o=Dict())
    ei = get(o, :epsinit,  EPS_INIT)
    ef = get(o, :epsfinal, EPS_FINAL)
    ed = get(o, :epsdecay, EPS_DECAY)
    et = ef + (ei-ef) * exp(-steps_done/ed)

    if rand() > et
        # @show w,s
        s = reshape(s,length(s),1)
        y = predict(w,s)
        y = Array(y)
        return indmax(y)
    else
        return rand(1:2)
    end
end

function train!(w, memory, opts; o=Dict())
    # get params
    gamma = get(o, :gamma, GAMMA)
    atype = get(o, :atype, Array{Float32})
    batchsize = get(o, :batchsize, BATCH_SIZE)

    # sample batch
    batch = sample(memory, batchsize)

    # prepare batch
    states  = mapreduce(i->i.state.data, hcat, batch)
    actions = map(i->i.action, batch)
    rewards = map(i->i.reward, batch)
    mask = map(i->!i.next_state.done, batch)
    mask = find(mask)
    next_states = filter(i->!i.next_state.done, batch)
    next_states = mapreduce(i->i.next_state.data, hcat, next_states)

    # convert batch
    states = convert(atype, states)
    rewards = convert(atype, rewards)
    next_states = convert(atype, next_states)

    # expected return
    y0 = predict(w,next_states)
    y1 = maximum(y0,1)
    y2 = sum(y0 .* (y0.==y1), 1)
    y3 = reshape(y2, 1, length(y2))
    expret = reshape(rewards, 1, length(rewards))
    expret[:,mask] += gamma * y3

    # training
    values = []
    g = gradient(w, states, actions, expret, mask; values=values)
    update!(w,g,opts)
    return values[1]
end

function objective(w, states, actions, expret, mask; values=[])
    qsa = predict(w, states)
    nrows,ncols = size(qsa)
    index = actions + nrows*(0:(length(actions)-1))
    qs  = qsa[index]
    qs  = reshape(qs, 1, length(qs))
    val = sumabs2(expret-qs) / length(qs)
    push!(values, val)
    return val
end

gradient = grad(objective)

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
