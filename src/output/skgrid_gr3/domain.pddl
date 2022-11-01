
(define (domain leogrid)
  (:requirements :typing )
  (:types thing location direction)
  (:predicates (move-dir ?v0 - location ?v1 - location ?v2 - direction)
	(clear ?v0 - location)
	(at ?v0 - thing ?v1 - location)
	(is-player ?v0 - thing)
	(move ?v0 - direction)
  )

  ; (:actions move)

  

	(:action move
		:parameters (?p - thing ?from - location ?to - location ?dir - direction)
		:precondition (and (move ?dir)
			(is-player ?p)
			(at ?p ?from)
			(clear ?to)
			(move-dir ?from ?to ?dir))
		:effect (and
			(not (at ?p ?from))
			(not (clear ?to))
			(at ?p ?to)
			(clear ?from))
	)
	

)
        