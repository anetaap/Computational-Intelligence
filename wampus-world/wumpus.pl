% general rules and initial state

:- dynamic breeze/2.
:- dynamic actor_pos/1.
:- dynamic pit/2.
:- dynamic gold_pos/1.
:- dynamic stench/2.
:- dynamic wumpos_pos/2.
:- dynamic wumpus_alive/0.

wumpus_pos(t(1, 1), no).
wumpus_alive.

actor_pos(t(1, 1)).

neigh(t(I, J), t(I2, J)) :-
    I > 1,
    I2 is I-1.
neigh(t(I, J), t(I2, J)) :-
    I < 4,
    I2 is I+1.
neigh(t(I, J), t(I, J2)) :-
    J > 1,
    J2 is J-1.
neigh(t(I, J), t(I, J2)) :-
    J < 4,
    J2 is J+1.

pit(t(1, 1), no).
pit(t(I, J), no) :-
    neigh(t(I, J), t(I2, J2)),
    breeze(t(I2, J2), no).

pit(t(I, J), yes) :- pit(t(I, J), yes, []).
pit(T, yes, Visited) :-
    neigh(T, TB),
    breeze(TB, yes),
    findall(TN, neigh(T, TN), Neighbors),
    subtract(Neighbors, TN, NotVisitedNeighbors),
    forall(pit(TN2, no, [TN2|Visited]), member(TN2, NotVisitedNeighbors)).
    

% our example

breeze(t(1, 1), no).
stench(t(1, 1), no).
gold_pos_hidden(t(2, 3)).
wumpus_pos_hidden(t(1, 3)).

action(move, T) :-
    actor_pos(P),
    neigh(P, T),
    retractall(actor_pos),
    assert(actor_pos(T)).

