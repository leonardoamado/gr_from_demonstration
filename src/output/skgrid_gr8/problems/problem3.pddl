
(define (problem p024-microban-sequential) (:domain leogrid)
  (:objects
        dir-down - direction
	dir-left - direction
	dir-right - direction
	dir-up - direction
	player-01 - thing
	pos-1-1 - location
	pos-1-2 - location
	pos-1-3 - location
	pos-1-4 - location
	pos-1-5 - location
	pos-1-6 - location
	pos-1-7 - location
	pos-2-1 - location
	pos-2-2 - location
	pos-2-3 - location
	pos-2-4 - location
	pos-2-5 - location
	pos-2-6 - location
	pos-2-7 - location
	pos-3-1 - location
	pos-3-2 - location
	pos-3-3 - location
	pos-3-4 - location
	pos-3-5 - location
	pos-3-6 - location
	pos-3-7 - location
	pos-4-1 - location
	pos-4-2 - location
	pos-4-3 - location
	pos-4-4 - location
	pos-4-5 - location
	pos-4-6 - location
	pos-4-7 - location
	pos-5-1 - location
	pos-5-2 - location
	pos-5-3 - location
	pos-5-4 - location
	pos-5-5 - location
	pos-5-6 - location
	pos-5-7 - location
	pos-6-1 - location
	pos-6-2 - location
	pos-6-3 - location
	pos-6-4 - location
	pos-6-5 - location
	pos-6-6 - location
	pos-6-7 - location
	pos-7-1 - location
	pos-7-2 - location
	pos-7-3 - location
	pos-7-4 - location
	pos-7-5 - location
	pos-7-6 - location
	pos-7-7 - location
  )
  (:goal (and
	(at player-01 pos-3-7)
  ))
  (:init 
	(at player-01 pos-7-7)

	(clear pos-1-1)
	(clear pos-1-2)
	(clear pos-1-3)
	(clear pos-1-4)
	(clear pos-1-5)
	(clear pos-1-6)
	(clear pos-1-7)
	(clear pos-2-7)
	(clear pos-3-1)
	(clear pos-3-2)
	(clear pos-3-3)
	(clear pos-3-4)
	(clear pos-3-5)
	(clear pos-3-6)
	(clear pos-3-7)
	(clear pos-4-1)
	(clear pos-4-2)
	(clear pos-4-3)
	(clear pos-4-4)
	(clear pos-4-5)
	(clear pos-4-6)
	(clear pos-4-7)
	(clear pos-5-1)
	(clear pos-5-2)
	(clear pos-5-3)
	(clear pos-5-4)
	(clear pos-5-5)
	(clear pos-5-6)
	(clear pos-5-7)
	(clear pos-6-1)
	(clear pos-7-1)
	(clear pos-7-2)
	(clear pos-7-3)
	(clear pos-7-4)
	(clear pos-7-5)
	(clear pos-7-6)
	(clear pos-7-7)
	(is-player player-01)

	(move dir-down)
	(move dir-left)
	(move dir-right)
	(move dir-up)
	(move-dir pos-1-1 pos-2-1 dir-right)
	(move-dir pos-1-1 pos-1-2 dir-down)

	(move-dir pos-1-2 pos-1-1 dir-up)
	(move-dir pos-1-2 pos-1-3 dir-down)
	(move-dir pos-1-2 pos-2-2 dir-right)

	(move-dir pos-1-3 pos-1-2 dir-up)
	(move-dir pos-1-3 pos-1-4 dir-down)
	(move-dir pos-1-3 pos-2-3 dir-right)

	(move-dir pos-1-4 pos-1-3 dir-up)
	(move-dir pos-1-4 pos-1-5 dir-down)
	(move-dir pos-1-4 pos-2-4 dir-right)

	(move-dir pos-1-5 pos-1-4 dir-up)
	(move-dir pos-1-5 pos-1-6 dir-down)
	(move-dir pos-1-5 pos-2-5 dir-right)

	(move-dir pos-1-6 pos-1-5 dir-up)
	(move-dir pos-1-6 pos-1-7 dir-down)
	(move-dir pos-1-6 pos-2-6 dir-right)

	(move-dir pos-1-7 pos-1-6 dir-up)
	(move-dir pos-1-7 pos-1-7 dir-down)
	(move-dir pos-1-7 pos-2-7 dir-right)

	(move-dir pos-2-1 pos-2-2 dir-down)
	(move-dir pos-2-1 pos-1-1 dir-left)
	(move-dir pos-2-1 pos-3-1 dir-right)

	(move-dir pos-2-2 pos-2-1 dir-up)
	(move-dir pos-2-2 pos-2-3 dir-down)
	(move-dir pos-2-2 pos-1-2 dir-left)
	(move-dir pos-2-2 pos-3-2 dir-right)

	(move-dir pos-2-3 pos-2-2 dir-up)
	(move-dir pos-2-3 pos-2-4 dir-down)
	(move-dir pos-2-3 pos-1-3 dir-left)
	(move-dir pos-2-3 pos-3-3 dir-right)

	(move-dir pos-2-4 pos-2-3 dir-up)
	(move-dir pos-2-4 pos-2-5 dir-down)
	(move-dir pos-2-4 pos-1-4 dir-left)
	(move-dir pos-2-4 pos-3-4 dir-right)
	
	(move-dir pos-2-5 pos-2-4 dir-up)
	(move-dir pos-2-5 pos-2-6 dir-down)
	(move-dir pos-2-5 pos-1-5 dir-left)
	(move-dir pos-2-5 pos-3-5 dir-right)
	
	(move-dir pos-2-6 pos-2-5 dir-up)
	(move-dir pos-2-6 pos-2-7 dir-down)
	(move-dir pos-2-6 pos-1-6 dir-left)
	(move-dir pos-2-6 pos-3-6 dir-right)

	(move-dir pos-2-7 pos-2-6 dir-up)
	(move-dir pos-2-7 pos-1-7 dir-left)
	(move-dir pos-2-7 pos-3-7 dir-right)

	(move-dir pos-3-1 pos-2-1 dir-left)
	(move-dir pos-3-1 pos-3-2 dir-down)
	(move-dir pos-3-1 pos-4-1 dir-right)

	(move-dir pos-3-2 pos-3-1 dir-up)
	(move-dir pos-3-2 pos-3-3 dir-down)
	(move-dir pos-3-2 pos-2-2 dir-left)
	(move-dir pos-3-2 pos-4-2 dir-right)

	(move-dir pos-3-3 pos-3-2 dir-up)
	(move-dir pos-3-3 pos-3-4 dir-down)
	(move-dir pos-3-3 pos-2-3 dir-left)
	(move-dir pos-3-3 pos-4-3 dir-right)

	(move-dir pos-3-4 pos-3-3 dir-up)
	(move-dir pos-3-4 pos-3-5 dir-down)
	(move-dir pos-3-4 pos-2-4 dir-left)
	(move-dir pos-3-4 pos-4-4 dir-right)

	(move-dir pos-3-5 pos-2-5 dir-left)
	(move-dir pos-3-5 pos-3-4 dir-up)
	(move-dir pos-3-5 pos-3-6 dir-down)
	(move-dir pos-3-5 pos-4-5 dir-right)

	(move-dir pos-3-6 pos-2-6 dir-left)
	(move-dir pos-3-6 pos-3-5 dir-up)
	(move-dir pos-3-6 pos-3-7 dir-down)
	(move-dir pos-3-6 pos-4-6 dir-right)
	
	(move-dir pos-3-7 pos-2-7 dir-left)
	(move-dir pos-3-7 pos-3-6 dir-up)
	(move-dir pos-3-7 pos-4-7 dir-right)

	(move-dir pos-4-1 pos-4-2 dir-down)
	(move-dir pos-4-1 pos-3-1 dir-left)
	(move-dir pos-4-1 pos-5-1 dir-right)
	
	(move-dir pos-4-2 pos-4-1 dir-up)
	(move-dir pos-4-2 pos-4-3 dir-down)
	(move-dir pos-4-2 pos-5-2 dir-right)
	(move-dir pos-4-2 pos-3-2 dir-left)

	(move-dir pos-4-3 pos-4-2 dir-up)
	(move-dir pos-4-3 pos-4-4 dir-down)
	(move-dir pos-4-3 pos-5-3 dir-right)
	(move-dir pos-4-3 pos-3-3 dir-left)

	(move-dir pos-4-4 pos-3-4 dir-left)
	(move-dir pos-4-4 pos-4-3 dir-up)
	(move-dir pos-4-4 pos-5-4 dir-right)
	(move-dir pos-4-4 pos-4-5 dir-down)

	(move-dir pos-4-5 pos-4-4 dir-up)
	(move-dir pos-4-5 pos-4-6 dir-down)
	(move-dir pos-4-5 pos-5-5 dir-right)
	(move-dir pos-4-5 pos-3-5 dir-left)

	(move-dir pos-4-6 pos-3-6 dir-left)
	(move-dir pos-4-6 pos-4-5 dir-up)
	(move-dir pos-4-6 pos-4-7 dir-down)
	(move-dir pos-4-6 pos-5-6 dir-right)

	(move-dir pos-4-7 pos-4-6 dir-up)
	(move-dir pos-4-7 pos-5-7 dir-right)
	(move-dir pos-4-7 pos-3-7 dir-left)

	(move-dir pos-5-1 pos-5-2 dir-down)
	(move-dir pos-5-1 pos-6-1 dir-right)
	(move-dir pos-5-1 pos-4-1 dir-left)

	(move-dir pos-5-2 pos-4-2 dir-left)
	(move-dir pos-5-2 pos-5-1 dir-up)
	(move-dir pos-5-2 pos-5-3 dir-down)
	(move-dir pos-5-2 pos-6-2 dir-right)

	(move-dir pos-5-3 pos-5-2 dir-up)
	(move-dir pos-5-3 pos-5-4 dir-down)
	(move-dir pos-5-3 pos-4-3 dir-left)
	(move-dir pos-5-3 pos-6-3 dir-right)

	(move-dir pos-5-4 pos-5-3 dir-up)
	(move-dir pos-5-4 pos-5-5 dir-down)
	(move-dir pos-5-4 pos-4-4 dir-left)
	(move-dir pos-5-4 pos-6-4 dir-right)
	
	(move-dir pos-5-5 pos-5-4 dir-up)
	(move-dir pos-5-5 pos-4-5 dir-left)
	(move-dir pos-5-5 pos-5-6 dir-down)
	(move-dir pos-5-5 pos-6-5 dir-right)

	(move-dir pos-5-6 pos-4-6 dir-left)
	(move-dir pos-5-6 pos-5-5 dir-up)
	(move-dir pos-5-6 pos-6-6 dir-right)
	(move-dir pos-5-6 pos-5-7 dir-down)

	(move-dir pos-5-7 pos-5-6 dir-up)
	(move-dir pos-5-7 pos-4-7 dir-left)
	(move-dir pos-5-7 pos-6-7 dir-right)
	
	(move-dir pos-6-1 pos-6-2 dir-down)
	(move-dir pos-6-1 pos-5-1 dir-left)
	(move-dir pos-6-1 pos-7-1 dir-right)
	
	(move-dir pos-6-2 pos-7-2 dir-right)
	(move-dir pos-6-2 pos-5-2 dir-left)
	(move-dir pos-6-2 pos-6-3 dir-down)
	(move-dir pos-6-2 pos-6-1 dir-up)

	(move-dir pos-6-3 pos-6-4 dir-down)
	(move-dir pos-6-3 pos-6-2 dir-up)
	(move-dir pos-6-3 pos-5-3 dir-left)
	(move-dir pos-6-3 pos-7-3 dir-right)

	(move-dir pos-6-4 pos-6-5 dir-down)
	(move-dir pos-6-4 pos-6-3 dir-up)
	(move-dir pos-6-4 pos-5-4 dir-left)
	(move-dir pos-6-4 pos-7-4 dir-right)

	(move-dir pos-6-5 pos-7-5 dir-right)
	(move-dir pos-6-5 pos-5-5 dir-left)
	(move-dir pos-6-5 pos-6-6 dir-down)
	(move-dir pos-6-5 pos-6-4 dir-up)

	(move-dir pos-6-6 pos-7-6 dir-right)
	(move-dir pos-6-6 pos-6-7 dir-down)
	(move-dir pos-6-6 pos-5-6 dir-left)
	(move-dir pos-6-6 pos-6-5 dir-up)

	(move-dir pos-6-7 pos-7-7 dir-right)
	(move-dir pos-6-7 pos-5-7 dir-left)
	(move-dir pos-6-7 pos-6-6 dir-up)

	(move-dir pos-7-1 pos-6-1 dir-left)
	(move-dir pos-7-1 pos-7-2 dir-down)

	(move-dir pos-7-2 pos-6-2 dir-left)
	(move-dir pos-7-2 pos-7-2 dir-down)
	(move-dir pos-7-2 pos-7-1 dir-up)

	(move-dir pos-7-3 pos-6-3 dir-left)
	(move-dir pos-7-3 pos-7-4 dir-down)
	(move-dir pos-7-3 pos-7-2 dir-up)

	(move-dir pos-7-4 pos-6-4 dir-left)
	(move-dir pos-7-4 pos-7-5 dir-down)
	(move-dir pos-7-4 pos-7-3 dir-up)

	(move-dir pos-7-5 pos-6-5 dir-left)
	(move-dir pos-7-5 pos-7-6 dir-down)
	(move-dir pos-7-5 pos-7-4 dir-up)

	(move-dir pos-7-6 pos-6-6 dir-left)
	(move-dir pos-7-6 pos-7-7 dir-down)
	(move-dir pos-7-6 pos-7-5 dir-up)

	(move-dir pos-7-7 pos-6-7 dir-left)
	(move-dir pos-7-7 pos-7-6 dir-up)
))
        