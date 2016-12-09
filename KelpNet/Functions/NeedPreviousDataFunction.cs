﻿using System;
using System.Collections.Generic;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousDataFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<NdArray[]> _prevInput = new List<NdArray[]>();
        private readonly List<NdArray[]> _prevOutput = new List<NdArray[]>();

        protected abstract NdArray NeedPreviousForward(NdArray x);
        protected abstract NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput);

        protected NeedPreviousDataFunction(string name, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
        }

        protected override NdArray ForwardSingle(NdArray x)
        {
            this._prevInput.Add(new[] { x });
            NdArray result = this.NeedPreviousForward(x);
            this._prevOutput.Add(new[] { result });

            return result;
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            this._prevInput.Add(x);

            NdArray[] prevoutput = new NdArray[x.Length];

#if DEBUG
            for(int i = 0; i < x.Length; i ++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                prevoutput[i] = this.NeedPreviousForward(x[i]);
            }
#if !DEBUG
            );
#endif

            this._prevOutput.Add(prevoutput);

            return prevoutput;
        }

        protected override NdArray BackwardSingle(NdArray gy)
        {
            NdArray prevInput = this._prevInput[this._prevInput.Count - 1][0];
            this._prevInput.RemoveAt(this._prevInput.Count - 1);

            NdArray prevOutput = this._prevOutput[this._prevOutput.Count - 1][0];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevInput, prevOutput);
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            NdArray[] prevInput = this._prevInput[this._prevInput.Count - 1];
            this._prevInput.RemoveAt(this._prevInput.Count - 1);

            NdArray[] prevOutput = this._prevOutput[this._prevOutput.Count - 1];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            NdArray[] result = new NdArray[gy.Length];

#if DEBUG
            for (int i = 0; i < gy.Length; i++)
#else
            Parallel.For(0, gy.Length, i =>
#endif
            {
                result[i] = this.NeedPreviousBackward(gy[i], prevInput[i], prevOutput[i]);
            }
#if !DEBUG
            );
#endif

            return result;
        }

        public override NdArray Predict(NdArray input)
        {
            return this.NeedPreviousForward(input);
        }

        public override NdArray[] Predict(NdArray[] x)
        {
            NdArray[] prevoutput = new NdArray[x.Length];
#if DEBUG
            for(int i = 0; i < x.Length; i ++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                prevoutput[i] = this.NeedPreviousForward(x[i]);
            }
#if !DEBUG
            );
#endif
            return prevoutput;
        }
    }
}
