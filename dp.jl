include(Pkg.dir("ReinforcementLearning", "examples", "maze","maze.jl"))

function main()
    maze_env = MazeEnv((5,5))
    discount=0.99

    print_maze(maze_env.maze)
    println("Start: $(maze_env.start)")
    println("Goal: $(maze_env.goal)")

    policy = Dict{MazeState, MazeAction}()

    # Initialize the policy with random actions
    for state in getAllStates(maze_env)
        policy[state] = rand(getActions(state, maze_env))
    end

    println("Initial Policy: ")
    print_policy(policy)

    V = policy_evaluation(maze_env, policy, discount)
    println("Initial State Values: ")
    print_state_values(V)

    opt_policy = policy_iteration(maze_env, policy, discount)
    println("Final Policy: ")
    print_policy(opt_policy)

    V = policy_evaluation(maze_env, opt_policy, discount)
    println("Final State Values: ")
    print_state_values(V)
end

function print_policy(policy::Dict{MazeState, MazeAction})
    # Print the policy by iterating over keys
    # MazeState -> MazeAction
    # e.g. MazeState(1,1) -> MazeAction(RIGHT)

    for (k,v) in policy
        println("$k -> $v")
    end
end

function print_state_values(V::Dict{MazeState, Float64})
    # Print the state values by iterating over keys
    # MazeState -> Float64
    # e.g. MazeState(1,1) -> 2.0

    for (k,v) in V
        println("$k -> $v")
    end
end

function policy_evaluation(env::MazeEnv, policy::Dict{MazeState, MazeAction}, discount=0.99)
    V = Dict{MazeState, Float64}() # State values
    # Initialize all state values with 0

    for state in getAllStates(env)
        V[state] = 0.0
    end

    eps = 1e-10

    # Evaluate the policy until there is no change between old values and new values
    # The maximum difference between values for a state must be less than eps
    while true
        delta = 0
        for state in getAllStates(env)
            temp = V[state]
            action = policy[state]
            successors = getSuccessors(state,action,env)
            V[state] = sum(map(i->i[3] * (i[2] + discount * V[i[1]]), successors))
            delta = max(delta, abs(temp-V[state]))
        end

        if delta < eps
            break
        end
    end

    return V
end

function policy_iteration(env::MazeEnv, policy::Dict{MazeState, MazeAction}, discount=0.99)
    # evaluate the policy
    # improve the policy
    # Implement the policy iteration algorithm

    while true
        V = policy_evaluation(env, policy, discount)
        policy_stable = true
        for state in getAllStates(env)
            temp = policy[state]
            x0 = map(a->(a,getSuccessors(state,a,env)), getActions(state,env))
            x1 = map(i->(i[1], map(j->j[3] * (j[2] + discount*V[j[1]]), i[2])), x0)
            x2 = map(i->(i[1],sum(i[2])), x1)
            x3 = sort(x2, by=i->i[2], rev=true)
            policy[state] = x3[1][1]
            if temp != policy[state]
                policy_stable = false
            end
        end

        if policy_stable
            break
        end
    end

    return policy
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
