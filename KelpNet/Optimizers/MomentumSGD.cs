﻿using KelpNet.Functions;

namespace KelpNet.Optimizers
{
    public class MomentumSGD : Optimizer
    {
        private double LearningRate;
        private double momentum;

        private NdArray[] vW;
        private NdArray[] vb;

        public MomentumSGD(FunctionStack fs, double learningRate = 0.01, double momentum = 0.9)
        {
            this.vW = new NdArray[fs.OptimizableFunctions.Count];
            this.vb = new NdArray[fs.OptimizableFunctions.Count];

            for (int i = 0; i < fs.OptimizableFunctions.Count; i++)
            {
                this.vW[i] = NdArray.ZerosLike(fs.OptimizableFunctions[i].W);

                if (fs.OptimizableFunctions[i].b != null)
                {
                    this.vb[i] = NdArray.ZerosLike(fs.OptimizableFunctions[i].b);
                }
            }

            this.LearningRate = learningRate;
            this.momentum = momentum;
        }

        protected override void Update(OptimizableFunction[] optimizableFunctions)
        {
            for (int i = 0; i < optimizableFunctions.Length; i++)
            {
                for (int j = 0; j < optimizableFunctions[i].W.Length; j++)
                {
                    vW[i].Data[j] *= this.momentum;
                    vW[i].Data[j] -= this.LearningRate * optimizableFunctions[i].gW.Data[j];

                    optimizableFunctions[i].W.Data[j] += vW[i].Data[j];
                }

                if (optimizableFunctions[i].b != null)
                {
                    for (int j = 0; j < optimizableFunctions[i].b.Length; j++)
                    {
                        vb[i].Data[j] *= this.momentum;
                        vb[i].Data[j] -= this.LearningRate * optimizableFunctions[i].gb.Data[j];

                        optimizableFunctions[i].b.Data[j] += vb[i].Data[j];
                    }
                }
            }
        }
    }
}
