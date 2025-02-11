(define (domain CAJAS_domain)
  (:requirements :strips)
  (:predicates
    (SOBRE ?x ?y)
    (LIBRE ?x)
    (SOSTENER ?x)
    (ENMESA ?x)
    (MANOLIBRE)
  )
(:action AGARRAR
    :parameters (?x)
    :precondition (and (LIBRE ?x) (ENMESA ?x) (MANOLIBRE))
    :effect (and (not (ENMESA ?x))
                 (not (LIBRE ?x))
                 (not (MANOLIBRE))
                 (SOSTENER ?x)))


(:action BAJAR
    :parameters (?x)
    :precondition (SOSTENER ?x)
    :effect (and (not (SOSTENER ?x))
                 (LIBRE ?x)
                 (MANOLIBRE)
                 (ENMESA ?x)))


(:action APILAR
    :parameters (?x ?y)
    :precondition (and (SOSTENER ?x) (LIBRE ?y))
    :effect (and (not (SOSTENER ?x))
                 (not (LIBRE ?y))
                 (LIBRE ?x)
                 (MANOLIBRE)
                 (SOBRE ?x ?y)))


(:action DESAPILAR
    :parameters (?x ?y)
    :precondition (and (SOBRE ?x ?y) (LIBRE ?x) (MANOLIBRE))
    :effect (and (SOSTENER ?x)
                 (LIBRE ?y)
                 (not (LIBRE ?x))
                 (not (MANOLIBRE))
                 (not (SOBRE ?x ?y))))

)