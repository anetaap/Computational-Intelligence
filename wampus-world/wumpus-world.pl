%------------------------------------------------------------------------------
% A Prolog Implementation of the Wumpus World described in
% Artificial Intelligence : A Modern Approach (Russel - Norvig)
%
% Mandatory Excercise 2007
% v1.0 - Jan. 31, 2007
% Richard O. Legendi
%
% Copied into prolog-examples with permission Richard O. Legendi
% Original exercise descriped in  Artificial Intelligence : A Modern Approach (Russel - Norvig)
%
% Usage:
% consult this file
% ?-start.
%
%------------------------------------------------------------------------------


%------------------------------------------------------------------------------
% Declaring dynamic methods

:- dynamic ([
         agent_location/1,
         gold_location/1,
         pit_location/1,
         time_taken/1,
         score/1,
         visited/1,
         visited_cells/1,
         world_size/1,
         wumpus_location/1,
             isPit/2,
             isWumpus/2,
             isGold/2
        ]).


%------------------------------------------------------------------------------
% To start the game

start :-
    format('Initializing started...~n', []),
    init,
    format('Let the game begin!~n', []),
    take_steps([[1,1]]).

%------------------------------------------------------------------------------
% Scheduling simulation:

step_pre(VisitedList) :-
    agent_location(AL),
    gold_location(GL),
    wumpus_location(WL),

    ( AL=GL -> score(S), time_taken(T), 
        format('_________________Result_________________~n', []),
        writeln('WON!'), format('Score: ~p,~nSteps: ~p', [S,T]),
        format('~n________________________________________~n', [])
    ; AL=WL -> 
        format('_________________Result_________________~n', []),
        format('Lost: Wumpus eats you!~n', []), update_score(-1000),
        score(S), time_taken(T), format('Score: ~p,~nSteps: ~p', [S,T]),
        format('~n________________________________________~n', [])
    ; is_pit(yes, AL) -> 
        format('_________________Result_________________~n', []),
        format('Lost: Fell into a pit!~n', []),update_score(-1000),
        score(S),time_taken(T), format('Score: ~p,~nSteps: ~p', [S,T]),
        format('~n________________________________________~n', [])
    ; take_steps(VisitedList)
    ).

take_steps(VisitedList) :-

    make_percept_sentence(Perception),
    agent_location(AL),
    format('I\'m in ~p, seeing: ~p~n', [AL,Perception]),
    
    update_KB(Perception),
    ask_KB(VisitedList, Action),

    format('I\'m going to: ~p~n', [Action]),

    update_time,
    update_score,

    agent_location(Aloc),
    VL = [Aloc|VisitedList],
    standing,
    step_pre(VL).

%------------------------------------------------------------------------------
% Updating states

update_time :-
    time_taken(T),
    NewTime is T+1,
    retractall( time_taken(_) ),
    assert( time_taken(NewTime) ).

update_score :-
    agent_location(AL),
    gold_location(GL),
    wumpus_location(WL),
    update_score(AL, GL, WL).

update_score(P) :-
    score(S),
    NewScore is S+P,
    retractall( score(_) ),
    assert( score(NewScore) ).

update_score(AL,GL,_) :-
    ( AL = GL, update_score(999));
    update_score(-1).

% score when agent meets the wumpus
update_score(AL,_,WP) :-
      AL = WP, update_score(-1000),
    fail.

% score when agent falls into a pit
update_score(AL,_,_) :-
    is_pit(yes, AL),
    update_score(-1000),
    fail.

update_agent_location(NewAL) :-
    retractall( agent_location(_) ),
    assert( agent_location(NewAL) ).

is_pit(no,  X) :-
    \+ pit_location(X).
is_pit(yes, X) :-
    pit_location(X).

%------------------------------------------------------------------------------
% Display standings

standing :-
    wumpus_location(WL),
    gold_location(GL),
    agent_location(AL),

    ( is_pit(yes, AL) -> (   format('Agent was fallen into a pit!~n'),  update_score(-1000))
     ;
    stnd(AL, GL, WL)
      %\+ pit_location(yes, Al),
    ).

stnd(_, _, _) :-
format('There\'s still something to do...~n', []).


stnd(AL, _, AL) :-
format('YIKES! You\'re eaten by the wumpus!', []), update_score(-1000),
fail.

stnd(AL, AL, _) :-
format('AGENT FOUND THE GOLD!!', []),
true.

% stnd(AL, _, _) :-
% (   is_pit(yes, AL),
% format('AGENT FELL INTO A PIT!!', []),
% fail).


%------------------------------------------------------------------------------
% Perceptotion

make_perception([_Stench,_Bleeze,_Glitter]) :-
    agent_location(AL),
    isStinky(AL),
    isBleezie(AL),
    isGlittering(AL).

test_perception :-
    make_percept_sentence(Percept),
    format('I feel ~p, ',[Percept]).

make_percept_sentence([Stench,Bleeze,Glitter]) :-
    smelly(Stench),
    bleezy(Bleeze),
    glittering(Glitter).

%------------------------------------------------------------------------------
% Initializing

init :-
    init_game,
    init_land_fig72,
    init_agent,
    init_wumpus,
    init_gold,
    init_pits,
    print_world.

init_game :-
    retractall( time_taken(_) ),
    assert( time_taken(0) ),

    retractall( score(_) ),
    assert( score(0) ),

    retractall( visited(_) ),
    assert( visited(1) ),

    retractall( isWumpus(_,_) ),
    retractall( isGold(_,_) ),

    retractall( visited_cells(_) ),
    assert( visited_cells([]) ).

% To set the situation described in Russel-Norvigs book (2nd Ed.),
% according to Figure 7.2
init_land_fig72 :-
    retractall( world_size(_) ),
    assert( world_size(4) ).

% function that initializes pits positions, pits are drown in each cell with probability 0.2 but not in the cell (1,1) 
init_pits :-
    retractall( pit_location(_) ),
    world_size(W),
    init_pits(W).

init_pits(0).
init_pits(W) :-
    init_pits_row(W),
    W1 is W-1,
    init_pits(W1).

init_pits_row(0).
init_pits_row(W) :-
    random(0, 10, R),
    (   R<2, W\=1 -> assert( pit_location([W,1]) ); true ),
    W1 is W-1,
    init_pits_row(W1).


% function that initializes gold location, gold can be everywhere except in the cell (1,1) and in the cell with wumpus
init_gold :-
    retractall( gold_location(_) ),
    world_size(W),
    random(1, W, X),
    random(1, W, Y),
    wumpus_location(WL),
    (   X\=1, Y\=1, [X,Y] \= WL -> assert( gold_location([X,Y]) );  init_gold).


% function that initializes agent location, agent is always in the cell (1,1)
init_agent :-
    retractall( agent_location(_) ),
    assert( agent_location([1,1]) ),

    visit([1,1]).

% function that initializes one wumpus, wumpus can be everywhere except in the cell (1,1)

init_wumpus :-
    retractall( wumpus_location(_) ),
    world_size(W),
    random(1, W, X),
    random(1, W, Y),( X\=1, Y\=1 -> assert( wumpus_location([X,Y]) ) ; init_wumpus ).

visit(Xs) :-
    visited_cells(Ys),
    retractall( visited_cells(_) ),
    assert( visited_cells([Ys|Xs]) ).
%------------------------------------------------------------------------------
% function that is printing the locations of pits, wumpus and gold

print_world :-
    wumpus_location(WL),
    gold_location(GL),
    findall(X, pit_location(X), PL),
    format('___________________World__________________~n'),
    format('Wumpus location: ~p,~nGold location: ~p,~nPits location: ~p~n', [WL,GL,PL]),
    format('__________________________________________~n').

%------------------------------------------------------------------------------
% Perceptors

%%% Institiation error!!!

%adj(X,Y) :-
%    world_size(WS),
%    ( X is Y+1, Y   < WS
%    ; X is Y-1, Y-1 > 0
%    ).

adj(1,2).
adj(2,1).
adj(2,3).
adj(3,2).
adj(3,4).
adj(4,3).

adjacent( [X1, Y1], [X2, Y2] ) :-
    ( X1 = X2, adj( Y1, Y2 )
    ; Y1 = Y2, adj( X1, X2 )
    ).

%adjacent([X1,Y],[X2,Y]) :-
%    adj(X1,X2).

%adjacent([X,Y1],[X,Y2]) :-
%    adj(Y1,Y2).

isSmelly(Ls1) :-
    wumpus_location( Ls2 ),
    adjacent( Ls1, Ls2 ).

isBleezy(Ls1) :-
    pit_location( Ls2 ),
    adjacent( Ls1, Ls2 ).

isGlittering( [X1, Y1] ) :-
    gold_location( [X2, Y2] ),
    X1 = X2,
    Y1 = Y2.

bleezy(yes) :-
    agent_location(AL),
    isBleezy(AL).

bleezy(no).

smelly(yes) :-
    agent_location(AL),
    isSmelly(AL).

smelly(no).

glittering(yes) :-
    agent_location(AL),
    isGlittering(AL).

glittering(no).

%------------------------------------------------------------------------------
% Knowledge Base:

update_KB( [Stench,Bleeze,Glitter] ) :-
    add_wumpus_KB(Stench),
    add_pit_KB(Bleeze),
    add_gold_KB(Glitter).

% if it would be 'yes' -> it would mean the player is eaten ;]
add_wumpus_KB(no) :-
    %agent_location(L1),
    %adjacent(L1, L2),
    %assume_wumpus(no, L2).
    agent_location([X,Y]),
    world_size(_),

    % Checking needed!!
    % adj will freeze for (4,_) !!

    Z1 is Y+1, assume_wumpus(no,[X,Z1]),
    Z2 is Y-1, assume_wumpus(no,[X,Z2]),
    Z3 is X+1, assume_wumpus(no,[Z3,Y]),
    Z4 is X-1, assume_wumpus(no,[Z4,Y]).

add_pit_KB(no) :-
    agent_location([X,Y]),
    Z1 is Y+1, assume_pit(no,[X,Z1]),
    Z2 is Y-1, assume_pit(no,[X,Z2]),
    Z3 is X+1, assume_pit(no,[Z3,Y]),
    Z4 is X-1, assume_pit(no,[Z4,Y]).

% Checking needed!! If its not already in the KB !!!
add_pit_KB(yes) :-
    agent_location([X,Y]),
    Z1 is Y+1, assume_pit(yes,[X,Z1]),
    Z2 is Y-1, assume_pit(yes,[X,Z2]),
    Z3 is X+1, assume_pit(yes,[Z3,Y]),
    Z4 is X-1, assume_pit(yes,[Z4,Y]).

add_gold_KB(no) :-
    gold_location(GL),
    assume_gold(no, GL).

add_gold_KB(yes) :-
    gold_location([X1,Y1]),
    agent_location([X2,Y2]),
    X1 = X2, Y1 = Y2,
    assume_gold(yes, [X1,Y1]).

assume_wumpus(no, L) :-
    retractall( isWumpus(_, L) ),
    assert( isWumpus(no, L) ),
    format('KB learn ~p - no Wumpus there!~n', [L]).

assume_wumpus(yes, L) :-
    %wumpus_healthy, % Will be included ...
    retractall( isWumpus(_, L) ),
    assert( isWumpus(yes, L) ),
    format('KB learn ~p - possibly the Wumpus is there!~n', [L]).

assume_pit(no, L) :-
    retractall( isPit(_, L) ),
    assert( isPit(no, L) ),
    format('KB learn ~p - there\'s no Pit there!~n', [L]).

assume_pit(yes, L) :-
    retractall( isPit(_, L) ),
    assert( isPit(yes, L) ),
    format('KB learn ~p - its a Pit!~n', [L]).

assume_gold(no, L) :-
    retractall( isGold(_, L) ),
    assert( isGold(no, L) ),
    format('KB learn ~p - there\'s no gold here!~n', [L]).

assume_gold(yes, L) :-
    retractall( isGold(_, L) ),
    assert( isGold(yes, L) ),
    format('KB learn ~p - GOT THE GOLD!!!~n', [L]).

permitted([X,Y]) :-
    world_size(WS),
    0 < X, X < WS+1,
    0 < Y, Y < WS+1,
    (agent_location([X1,Y1]),
    adjacent([X1,Y1],[X,Y])).

ask_KB(VisitedList, Action) :-
    isWumpus(no, L),
    isPit(no, L),
    permitted(L),
    not_member(L, VisitedList),
    update_agent_location(L),
    Action = L.

%------------------------------------------------------------------------------
% Utils

not_member(_, []).
not_member([X,Y], [[U,V]|Ys]) :-
    ( X=U,Y=V -> fail
    ; not_member([X,Y], Ys)
    ).