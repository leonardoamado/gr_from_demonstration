(define (problem pb1) (:domain logistics)
(:objects A B C D
)

(:init
	(connected a b)
	(connected b a)
	(connected b c)
	(connected c b)
	(connected c d)
	(connected d c)
	(package c)
	(truckpos a)

; action literals
	(drive c b)
	(drive c d)
	(drive b c)
	(drive b a)
	(drive a b)
	(drive d c)
	(load c)
	(load b)
	(load a)
	(load d)
	(load truck)
	(unload c)
	(unload b)
	(unload a)
	(unload d)
	(unload truck)
	)

(:goal (and
	(ONTABLE D), (ON C D), (CLEAR C)))
)
