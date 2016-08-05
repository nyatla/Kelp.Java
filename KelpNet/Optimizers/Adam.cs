﻿using System;
using System.Collections.Generic;
using KelpNet.Functions;

namespace KelpNet.Optimizers
{
    public class Adam : Optimizer
    {
        private double alpha;
        private double beta1;
        private double beta2;
        private double eps;

        double lr
        {
            get
            {
                double fix1 = 1 - Math.Pow(this.beta1, this.t);
                double fix2 = 1 - Math.Pow(this.beta2, this.t);
                return this.alpha * Math.Sqrt(fix2) / fix1;
            }
        }

        private NdArray[][] m;
        private NdArray[][] v;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        {
            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
        }

        protected override void DoUpdate(List<OptimizableFunction> optimizableFunctions)
        {
            for (int i = 0; i < optimizableFunctions.Count; i++)
            {
                for (int j = 0; j < optimizableFunctions[i].Parameters.Count; j++)
                {
                    for (int k = 0; k < optimizableFunctions[i].Parameters[j].Length; k++)
                    {
                        double grad = optimizableFunctions[i].Parameters[j].Grad.Data[k];

                        m[i][j].Data[k] += (1 - this.beta1) * (grad - m[i][j].Data[k]);
                        v[i][j].Data[k] += (1 - this.beta2) * (grad * grad - v[i][j].Data[k]);

                        optimizableFunctions[i].Parameters[j].Param.Data[k] -= lr * m[i][j].Data[k] / (Math.Sqrt(v[i][j].Data[k]) + this.eps);
                    }
                }
            }
        }

        public override void Initialize(FunctionStack fs)
        {
            this.m = new NdArray[fs.OptimizableFunctions.Count][];
            this.v = new NdArray[fs.OptimizableFunctions.Count][];

            for (int i = 0; i < fs.OptimizableFunctions.Count; i++)
            {
                this.m[i] = new NdArray[fs.OptimizableFunctions[i].Parameters.Count];
                this.v[i] = new NdArray[fs.OptimizableFunctions[i].Parameters.Count];

                for (int j = 0; j < fs.OptimizableFunctions[i].Parameters.Count; j++)
                {
                    this.m[i][j] = NdArray.ZerosLike(fs.OptimizableFunctions[i].Parameters[j].Param);
                    this.v[i][j] = NdArray.ZerosLike(fs.OptimizableFunctions[i].Parameters[j].Param);
                }
            }
        }
    }
}