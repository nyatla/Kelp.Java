﻿using System;

namespace KelpNet.Functions.Activations
{
    public class Sigmoid : Function, IPredictableFunction
    {
        public override NdArray Forward(NdArray x)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = 1 / (1 + Math.Exp(-x.Data[i]));
            }

            return y;
        }

        public override NdArray Backward(NdArray gy, NdArray PrevInput, NdArray PrevOutput)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = gy.Data[i] * PrevOutput.Data[i] * (1 - PrevOutput.Data[i]);
            }

            return result;
        }

        public NdArray Predict(NdArray input)
        {
            return Forward(input);
        }
    }
}