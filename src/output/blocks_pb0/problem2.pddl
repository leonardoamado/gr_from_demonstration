(define (problem pb1)
    (:domain blocks)
    (:objects d r a w o )
    (:init
	(handempty)
	(clear r)
	(ontable a)
	(on r w)
	(on w o)
	(ontable d)
	(on o d)
	(clear a)

; action literals
	(pickup o)
	(pickup a)
	(pickup d)
	(pickup w)
	(pickup r)
	(putdown o)
	(putdown a)
	(putdown d)
	(putdown w)
	(putdown r)
	(stack o o)
	(stack o a)
	(stack o d)
	(stack o w)
	(stack o r)
	(stack a o)
	(stack a a)
	(stack a d)
	(stack a w)
	(stack a r)
	(stack d o)
	(stack d a)
	(stack d d)
	(stack d w)
	(stack d r)
	(stack w o)
	(stack w a)
	(stack w d)
	(stack w w)
	(stack w r)
	(stack r o)
	(stack r a)
	(stack r d)
	(stack r w)
	(stack r r)
	(unstack o o)
	(unstack o a)
	(unstack o d)
	(unstack o w)
	(unstack o r)
	(unstack a o)
	(unstack a a)
	(unstack a d)
	(unstack a w)
	(unstack a r)
	(unstack d o)
	(unstack d a)
	(unstack d d)
	(unstack d w)
	(unstack d r)
	(unstack w o)
	(unstack w a)
	(unstack w d)
	(unstack w w)
	(unstack w r)
	(unstack r o)
	(unstack r a)
	(unstack r d)
	(unstack r w)
	(unstack r r)
	    )
    (:goal
        (and
	(on a w),(handempty),(ontable w),(clear d),(ontable o),(clear o),(on d r),(on r a)
        )
    )
)