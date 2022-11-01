(define (domain blocks2)
  (:requirements :strips)
  
(:predicates (clear ?x)
             (ontable ?x)
             (handempty)
             (holding ?x)
             (on ?x ?y)
             (pickup ?x)
             (putdown ?x)
             (stack ?x ?y)
             (unstack ?x ?y))
             
  ; (:actions putdown stack unstack pickup)

(:action pick-up
  :parameters (?x)
  :precondition (and (clear ?x) (ontable ?x) (handempty))
  :effect (and (holding ?x) (not (clear ?x)) (not (ontable ?x))
               (not (handempty))))

(:action put-down
  :parameters  (?x)
  :precondition (holding ?x)
  :effect (and (clear ?x) (handempty) (ontable ?x)
               (not (holding ?x))))

(:action stack
  :parameters (?x ?y)
  :precondition (and (clear ?y) (holding ?x))
  :effect (and (handempty) (clear ?x) (on ?x ?y)
               (not (clear ?y)) (not (holding ?x))))

(:action unstack
  :parameters  (?x ?y)
  :precondition (and (on ?x ?y) (clear ?x) (handempty))
  :effect (and (holding ?x) (clear ?y)
               (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))))
