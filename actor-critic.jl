using Knet
using ReinforcementLearning
using ArgParse
using JLD

const GAMMA = 0.95
const WINIT = 0.1
const EPISODES = 1000

function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        ("--actor"; nargs='*'; arg_type=Int64)
        ("--critic"; nargs='*'; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}":"Array{Float32}"))
        ("--optim"; default="Sgd(;lr=0.001)")
        ("--discount"; default=GAMMA; arg_type=Float64)
        ("--winit"; default=WINIT; arg_type=Float64)
        ("--loadfile"; default=nothing)
        ("--episodes"; default=EPISODES; arg_type=Int64)
        ("--seed"; default=-1; arg_type=Int64)
        ("--monitor"; default=nothing)
        ("--generate"; default=0; arg_type=Int64)
        ("--period"; default=100; arg_type=Int64)
        ("--pole"; default=1; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()

    pole = o[:pole]
    threshold = pole == 0 ? 200 : 500
    env = GymEnv("CartPole-v$pole")
    w = θ = opts = nothing
    if o[:loadfile] == nothing
        θ = initweights(o[:atype], o[:actor])
        w = initweights(o[:atype], o[:critic])
    else
        w = load(o[:loadfile], "w")
        w = map(wi->convert(o[:atype]), w)
        θ = load(o[:loadfile], "w")
        θ = map(θi->convert(o[:atype]), θ)
    end
    aopt = map(θi->eval(parse(o[:optim])), θ)
    copt = map(wi->eval(parse(o[:optim])), w)

    if o[:monitor] != nothing
        monitor_start(env, o[:monitor])
    end

    best_episode = -Inf
    avgreward = 0
    for k = 1:o[:episodes]
        s0 = getInitialState(env); state = s0;
        episode_rewards = 0
        episode_length  = 0
        history = []

        while !state.done
            s = convert(o[:atype], state.data)
            probs = softmax(predict(θ,s))

            # this is a, action
            a = action = sample_action(probs)[1]

            # this is transition sampled
            next_state, reward = transfer(env, state, env.actions[action])
            push!(history, Transition(state, action, next_state, reward))

            # compute TD target
            sp = history[end].state.data
            ap = sample_action(softmax(predict(θ,s)))[1]
            td = reward + o[:discount] * predict(w,sp)[ap]

            if next_state.done && episode_length < threshold
                td -= threshold
            end

            values = []
            s = convert(o[:atype], s)
            ∇w = criticgrad(w, s, a, td; values=values)
            qw = values[1]

            # train actor
            ∇θ = actorgrad(θ, s, a, qw)
            update!(θ, ∇θ, aopt)

            # train critic
            update!(w, ∇w, copt)

            state = next_state
            episode_length += 1
            episode_rewards += reward
        end

        # average_reward = episode_rewards / episode_length
        if k < 100
            avgreward = avgreward * (k-1) + episode_rewards
            avgreward = avgreward / k
        else
            avgreward = 0.01 * episode_rewards + avgreward * 0.99
        end

        if k % o[:period] == 0
            println("(episode:$k,avgreward:$avgreward)")
        end

        if avgreward > best_episode
            best_episode = avgreward
        end

        empty!(history)
    end

    for k = 1:o[:generate]
        done = false
        state = getInitialState(env)
        rewards = 0
        while !done
            s = convert(o[:atype], state.data)
            pred = predict(θ,s)
            probs = softmax(pred)
            # action = sample_action(probs)[1]
            action = select_action(probs)
            render_env(env)
            next_state, reward = transfer(env,state,env.actions[action])
            state = next_state
            done = next_state.done
            rewards += reward
        end
        println("(generation:$k,reward:$rewards")
    end
end

type Transition
    state
    action
    next_state
    reward
end

function initweights(atype, hidden, xsize=4, ysize=2, winit=0.1)
    w = []
    x = xsize
    for y in [hidden..., 2]
        push!(w, xavier(y,x))
        push!(w, zeros(y,1))
        x = y
    end
    return map(wi->convert(atype,wi), w)
end

function predict(w,x)
    for i = 1:2:length(w)
        x = w[i] * x .+ w[i+1]
        if i < length(w)-1
            x = relu(x)
        end
    end
    return x
end

function softmax(x0)
    x1 = maximum(x0,1)
    x2 = x0 .- x1
    x3 = exp(x2)
    x4 = x3 ./ sum(x3,1)
end

function actor(w, x, actions, values)
    ypred = predict(w,x)
    return -sum(logprob(actions,ypred) .* values) / size(x,2)
end

actorgrad = grad(actor)

function critic(w,s,a,ret; values=[])
    qsa = predict(w,s)[a]
    val = sumabs2(ret-qsa)
    push!(values, val)
    return val
end

criticgrad = grad(critic)

function sample_action(probs)
    probs = Array(probs)

    cprobs = cumsum(probs)
    sampled = cprobs .> rand()
    actions = mapslices(indmax, sampled, 1)

    # an alternative?
    # sampled = probs .* rand(size(probs))
    # actions = mapslices(indmax, sampled, 1)

    return actions
end

function select_action(probs)
    return indmax(probs)
end

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    o1 = logp(ypred,1)
    o2 = o1[index]
    return o2
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
