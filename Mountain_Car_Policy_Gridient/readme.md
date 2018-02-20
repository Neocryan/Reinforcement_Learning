
Exercise session 4: Policy gradient

Guillaume Charpiat, Victor Berger January 25, 2018

\

Preamble 
========

This exercise session is a followup of the previous one.

\

Exercise
========

In this project, you are asked to solve a variation of the mountain car
problem, using a policy gradient algorithm.

We are considering an environment similar to the one of last week. Once
again, this is a mountain car problem, however we have changed some
parts of the problem:

−

−

-   The possible actions for your car are now continuous: any force
    value in \[ [F]{.s4}[max]{.s5}; [F]{.s4}[max]{.s5}\] (if you output
    values outside of this range they will be clamped).

    • − − | |

    • − − | |

    You receive at each time step a reward of 0[.]{.s4}1 [λ F]{.s4}[t
    ]{.s5}[2]{.s6}, where [F]{.s4}[t ]{.s5}is the force you last
    applied.

    •

    •

    λ [and ]{.p}F[max ]{.s5}[are parameters of the environment that you
    do not know in advance (and will be randomized on the test
    bed).]{.p}

-   You receive a reward of 100 for reaching the top of the hill.

    This means your agent must find an appropriate balance between
    getting out of the valley as quickly as possible and not
    accelerating too much (to save fuel).

    Given the action space is now continuous, using an approximate
    Q-learning algorithm like last time becomes difficult. A possibility
    would be to discretize the force values, but this does not combine
    well with the fact that [F]{.s4}[max ]{.s5}and [λ ]{.s4}are unknown.
    Instead, we will use a policy gradient method, which can easily
    model continuous action spaces.

    \

    \

    We suggest the following model:

-   [The policy ]{.p}π[θ]{.s5}[t]{.s7}[
    ]{.s8}[(]{.p}a[t]{.s5}[|]{.s3}s[t]{.s5}[) is always a normal
    distribution, of mean ]{.p}µ[t]{.s5}[ ]{.s9}[and standard deviation
    ]{.p}σ[t]{.s5}[;]{.p}

    µ[t]{.s5}[ ]{.s9}[= ]{.p}µ[θ]{.s5}[ ]{.s9}[(]{.p}s[t]{.s5}[)
    =]{.p}[t]{.s7}

    µ[t]{.s5}[ ]{.s9}[= ]{.p}µ[θ]{.s5}[ ]{.s9}[(]{.p}s[t]{.s5}[)
    =]{.p}[t]{.s7}

    θ[t]{.s9}

    i,j

    θ[t]{.s9}

    i,j

    φ[i,j]{.s5}[ ]{.s9}[(]{.p}s[t]{.s5}[),]{.p}

    φ[i,j]{.s5}[ ]{.s9}[(]{.p}s[t]{.s5}[),]{.p}

-   [µ]{.s4}[t ]{.s5}is a linear fu [了]{.s11}nction of pre-defined
    descriptors of the state:

    i,j

    i,j

    \

    i,j

    i,j

    where [φ]{.s4}[i,j ]{.s5}is the same kernel basis as for previous
    exercise;

-   σ[t]{.s5}[ ]{.s9}[is annealed: it starts with a rather high value
    (to encourage exploration) and decreases over time as the model
    ]{.p}µ[θ]{.s5}[t]{.s7}[ ]{.s8}[gets better.]{.p}

    (The following questions are here to guide you though the
    implementation, and do not require a written answer.)

    \

    [Question 1: ]{.h2}[What is the expression of log
    ]{.p}[π]{.s4}θ[t]{.s7}[ ]{.s8}[(]{.p}[a]{.s4}t[|]{.s3}[s]{.s4}t[) ?
    Of ]{.p}[∇]{.s3}θ[ ]{.s9}[log ]{.p}[π]{.s4}θ[
    ]{.s9}[(]{.p}[a]{.s4}t[|]{.s3}[s]{.s4}t[) ?]{.p}

    We consider implementing the actor-critic policy gradient algorithm
    to solve this problem. For this, we also need an estimation of the
    value function [V ]{.s4}([s]{.s4}[t]{.s5}). We will learn it using a
    linear parametric model as well:

    t

    t

    i,j

    i,j

    V[ψ ]{.s5}[(]{.p}s[t]{.s5}[) = ]{.p}[「]{.s12}ψ[t]{.s13}

    i,j

    i,j

    \

    φ[i,j ]{.s5}[(]{.p}s[t]{.s5}[)]{.p}

    \

    i,j

    i,j

    \

    **Question 2:** What is the update rule for the parameters
    [ψ]{.s4}[i,j ]{.s5}of [V]{.s4}[ψ ]{.s5}?

    \

    **Question 3:** What is the update rule for the parameters
    [θ]{.s4}[i,j ]{.s5}of [π]{.s4}[θ ]{.s5}?

    \

    **Question 4:** How does the value of [σ ]{.s4}impact the update of
    [θ]{.s4}[i,j ]{.s5}? Design an annealing rule for [σ
    ]{.s4}appropriately.

    \

    **Question 5:** Seeing the update rule for [θ]{.s4}[i,j ]{.s5},
    would it make sense to do an [c]{.s4}-greedy-like version of this
    algorithm (acting completely randomly with probability [c]{.s4})?

    \

    Question 6: [Then, how can we go into ”exploitation mode” with this policy? What else should we take care of when doing so?]
    --------------------------------------------------------------------------------------------------------------------------------

    \

    Implement the actor-critic policy gradient for this problem and
    submit your solution on the platform.

    \

    Template description
    ====================

    The template is a zip file that you can download on the course
    website. It contains several files, two of them are of interest for
    you: [agent.py ]{.s14}and [main.py]{.s14}. [agent.py ]{.s14}is the
    file in which you will write the code of your agent, using the
    [RandomAgent ]{.s14}class as a template. Don’t forget to read the
    documentation

    \

    \

    it contains. In particular, note that for this exercise, at the
    beginning of a new game, the [reset ]{.s14}function called returns
    to the agent information about the range of possible [x
    ]{.s4}coordinates. As usual you can have the code of your several
    agents in the same file, and use the final line [Agent = MyAgent
    ]{.s14}to choose which agent you want to run.

    main.py [is the program that will actually run your agent. You can
    run it with the command ]{.p}python main.py[. It also accepts a few
    command-line arguments:]{.p}

    •

    •

    --ngames N [will run your agent for N games against in the same
    environ- ment and report the total cumulative reward]{.p}

-   --niter N [maximum number of iterations allowed for each game]{.p}

    •

    •

    --batch B [will run B instances of your agent in parallel, each
    against its own bandit, and report the average total cumulative
    reward]{.p}

    •

    •

    --verbose [will print details at each step of what your agent did.
    This can be helpful to understand if something is going wrong.]{.p}

    •

    •

    --interactive [will train your agent ]{.p}ngames [times, then run an
    interac- tive game displaying informational plots. You need to have
    ]{.p}matplotlib [installed to use it.]{.p}

    The running of your agent follows a general procedure that will be
    shared for all later practicals:

-   The environment generates an observation

    •

    •

    This observation is provided to your agent via the [act
    ]{.s14}method which chooses an action

-   The environment processes your action to generate a reward

