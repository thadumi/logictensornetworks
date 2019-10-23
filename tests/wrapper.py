#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import unittest

from ltn import logictensornetworks_wrapper as ltnw


class TestBasics(unittest.TestCase):
    def setUp(self):
        ltnw._reset()

    def testConstant(self):
        zero=ltnw.constant("zero",[0,0])
        one=ltnw.constant("one",[1,1])
        
        self.assertEqual(ltnw.constant("zero"),zero)        
        self.assertEqual(ltnw.constant("one"),one)

    def testVariable(self):
        v1=ltnw.variable("?var1",[1,2,3])
        v2=ltnw.variable("?var2",2)
        self.assertEqual(ltnw.variable("?var1"),v1)
        self.assertEqual(ltnw.variable("?var2"),v2)
        
    def testPredicate(self):
        p1=ltnw.predicate("P1",2)
        p2=ltnw.predicate("P2",2)
        self.assertEqual(ltnw.predicate("P1"),p1)
        self.assertEqual(ltnw.predicate("P2"),p2)
        self.assertNotEqual(ltnw.predicate("P1"),p2)


    def testFunction(self):
        f1=ltnw.function("F1",2)
        f2=ltnw.function("F2",2)
        self.assertEqual(ltnw.function("F1"),f1)
        self.assertEqual(ltnw.function("F2"),f2)

    def testParseTerm(self):
        self.assertEqual(ltnw._parse_term("a"),"a")
        self.assertEqual(ltnw._parse_term("f(c)"),['f', ['c']])
        
    def testBuildTerm(self):        
        data = numpy.random.uniform([-1,-1],[1,1],(500,2),).astype(numpy.float32)
        ltnw.constant("c",2)
        ltnw.variable("?var",data)
        ltnw.function("f",2,fun_definition=lambda d:d)
        ltnw.function("g",4,fun_definition=lambda d:d)
        
        self.assertEqual(ltnw.constant("c"), ltnw._build_term('c'))
        
        self.assertEqual(ltnw.variable("?var"), ltnw._build_term('?var'))

        self.assertIsNotNone(ltnw._build_term(['f', ['?var']]))
        self.assertIsNotNone(ltnw._build_term(['f', [['f', ['?var']]]]))

        with self.assertRaises(Exception):
            ltnw._build_term(['h', ['?var']]) # h not declared
        with self.assertRaises(Exception):
            self.assertRaises(ltnw._build_term(['g', ['?vars']])) # vars not declared

    def testParseFormula(self):
        self.assertEqual(ltnw._parse_formula("P(a)"),["P",["a"]])
        self.assertEqual(ltnw._parse_formula("P(a,b)"),["P",["a","b"]])
        self.assertEqual(ltnw._parse_formula("P(f(a))"),["P",[["f",["a"]]]])
        self.assertEqual(ltnw._parse_formula("P(f(a),b)"),["P",[["f",["a"]],"b"]])
        self.assertEqual(ltnw._parse_formula("~P(a)"),['~', ['P', ['a']]])        
        self.assertEqual(ltnw._parse_formula("P(a) & P(b)"),[['P', ['a']], '&', ['P', ['b']]])
        self.assertEqual(ltnw._parse_formula("P(a) | P(b)"),[['P', ['a']], '|', ['P', ['b']]])
        self.assertEqual(ltnw._parse_formula("P(a) -> P(b)"),[['P', ['a']], '->', ['P', ['b']]])
        self.assertEqual(ltnw._parse_formula("forall ?a: P(?a)"),['forall', '?a', ['P', ['?a']]])
        self.assertEqual(ltnw._parse_formula("exists ?a: P(?a)"),['exists', '?a', ['P', ['?a']]])

    def testBuildFormula(self):
        data = numpy.random.uniform([-1,-1],[1,1],(500,2),).astype(numpy.float32)
        ltnw.constant("c",[1.,0])
        ltnw.variable("?var",data)
        ltnw.variable("?var2",data)
        ltnw.function("f",2,fun_definition=lambda d:d[:,:2])
        ltnw.function("g",4,fun_definition=lambda d:d)
        ltnw.predicate("P",2)
        ltnw.predicate("B",2)
        ltnw.predicate("REL",4)
        
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(c)")))
        with self.assertRaises(Exception):
            ltnw._build_formula(ltnw._parse_formula("P(d)"))
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(?var)")))
        with self.assertRaises(Exception):
            ltnw._build_formula(ltnw._parse_formula("P(?vars)"))
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(f(?var))")))
        with self.assertRaises(Exception):
            ltnw._build_formula(ltnw._parse_formula("P(h(?var))")) # h not declared
        
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("~P(?var)")))
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("~P(f(?var))")))
        
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("~REL(?var,?var2)")))
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("~REL(?var,f(g(?var2)))")))
        with self.assertRaises(Exception):
            self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("~REL(f(?var))")))

        for op in ["&","|","->"]:
            self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(?var) %s ~ P(?var)" % op)))
            self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(?var) %s ~ P(?var)" % op)))
            self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("~P(?var) %s P(?var)" % op)))
            self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("~P(?var) %s ~P(?var)" % op)))

        for i in range(10):
            self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(?var) %s ~P(?var)%s ~P(?var)%s P(?var)" % tuple(numpy.random.permutation(["&","|","->"])))))

        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("forall ?var: P(?var) & ~ P(?var)")))
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("forall ?var,?var2: P(?var) & ~ P(?var2)")))        
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(c) & (forall ?var,?var2: P(?var) & ~ P(?var2))")))

        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("exists ?var: P(?var) & ~ P(?var)")))
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("exists ?var,?var2: P(?var) & ~ P(?var2)")))        


        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("forall ?var: (exists ?var2: P(?var) & ~ P(?var2))")))
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("forall ?var: (exists ?var2: P(?var) & ~ P(?var2))")))
        
        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("(forall ?var: (exists ?var2: (P(?var) & P(?var2) & (forall ?var: P(?var)))))")))

        self.assertIsNotNone(ltnw._build_formula(ltnw._parse_formula("P(c) | P(?var)")))

class TestSimpleExperiments(unittest.TestCase):
    def setUp(self):
        ltnw._reset()

    def testSimplePredicate(self):
        import tensorflow
        
        nr_samples=100

        ltnw.constant("a",[2.,3.])
        ltnw.variable("?data_A",numpy.random.uniform([0.,0.],[.1,1.],(nr_samples,2)).astype("float32"))
        
        mu=tensorflow.constant([2.,3.])
        ltnw.predicate("A",2,pred_definition=lambda x: tensorflow.exp(-tensorflow.norm(tensorflow.subtract(x,mu),axis=1)));       
        
        self.assertEqual(ltnw.ask("A(a)"),1.)
        self.assertGreater(ltnw.ask("forall ?data_A: A(?data_A)"),0.)
        
    def testSimplePredicateOptimization(self):
        nr_samples=100

        ltnw.variable("?data_A",numpy.random.uniform([0.,0.],[.1,1.],(nr_samples,2)).astype("float32"))
        ltnw.variable("?data_not_A",numpy.random.uniform([2.,0],[3.,1.],(nr_samples,2)).astype("float32"))
        
        ltnw.predicate("A",2)
        
        ltnw.axiom("forall ?data_A: A(?data_A)")
        ltnw.axiom("forall ?data_not_A: ~A(?data_not_A)")
        
        ltnw.initialize_knowledgebase(initial_sat_level_threshold=.1)
        sat_level=ltnw.train(track_sat_levels=10000,sat_level_epsilon=.99)
        
        self.assertGreater(sat_level,.8)
            
        ltnw.constant("a",[0.5,0.5])
        ltnw.constant("b",[2.5,0.5])
        
        self.assertGreater(ltnw.ask("A(a)")[0],.8)
        self.assertGreater(ltnw.ask("~A(b)")[0],.8)

        result=ltnw.ask_m(["A(a)","~A(b)"])
        
        for r in result:
            self.assertGreater(r[0],.8)
            self.assertGreater(r[0],.8)

if __name__ == "__main__":
    unittest.main()
