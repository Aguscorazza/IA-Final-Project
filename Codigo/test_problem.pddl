(define 
(problem BLOCKS-4)
(:domain STRIPS_domain)
(:objects WASHER NUT NAIL BOLT)
(:init (HANDEMPTY)
(CLEAR BOLT)
(ONTABLE NAIL)
(ON BOLT WASHER)
(ON WASHER NUT)
(ON NUT NAIL)
)
(:goal (and (HANDEMPTY)
(CLEAR NUT)
(ONTABLE NAIL)
(ON NUT WASHER)
(ON WASHER BOLT)
(ON BOLT NAIL)
))
)
