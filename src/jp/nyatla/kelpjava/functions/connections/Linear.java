package jp.nyatla.kelpjava.functions.connections;

import jp.nyatla.kelpjava.OptimizeParameter;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.NeedPreviousInputFunction;

//    [Serializable]
    final public class Linear extends NeedPreviousInputFunction
    {
        public final NdArray W;
        public final NdArray b;
        public final NdArray gW;
        public final NdArray gb;
        
        /**
         * コピーコンストラクタ
         * @param i_src
         */
        public Linear(Linear i_src)
        {
        	super(i_src);
        	this.W=(NdArray) i_src.W.deepCopy();
        	this.b=(NdArray) i_src.b.deepCopy();
        	this.gW=(NdArray) i_src.gW.deepCopy();
        	this.gb=(NdArray) i_src.gb.deepCopy();
        }
        public Linear(int i_inputCount, int i_outputCount)
        {
        	this(i_inputCount,i_outputCount,false,null,null,"Linear");
        }
        public Linear(int i_inputCount, int i_outputCount,String i_name)
        {
        	this(i_inputCount,i_outputCount,false,null,null,i_name);
        }
        
        public Linear(int i_inputCount, int i_outputCount, boolean noBias,double[] initialW, double[] initialb, String i_name)
        {
        	super(i_name, i_inputCount, i_outputCount);
            this.W = NdArray.zeros(i_outputCount,i_inputCount);
            this.gW = NdArray.zerosLike(this.W);
            this.parameters = new OptimizeParameter[noBias ? 1 : 2];
            if (initialW == null)
            {
                this.initWeight(this.W);
            }
            else
            {
            	System.arraycopy(initialW, 0,this.W.data,0,initialW.length);
            }
            this.parameters[0] = new OptimizeParameter(this.W, this.gW, this.name + " W");
            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.zeros(i_outputCount);
            this.gb = NdArray.zerosLike(this.b);
            if (!noBias)
            {
                if (initialb != null)
                {
                	System.arraycopy(initialb, 0,this.b.data,0,initialb.length);
                }
                this.parameters[1] = new OptimizeParameter(this.b, this.gb, this.name + " b");
            }
        }

        @Override
        protected NdArray needPreviousForward(NdArray i_x)
        {
            //バイアスを最初から入れておく
            double[] output = this.b.data.clone();

            for (int i = 0; i < this.outputCount; i++)
            {
                int indexOffset = this.inputCount * i;

                for (int j = 0; j < this.inputCount; j++)
                {
                    output[i] += i_x.data[j] * this.W.data[indexOffset + j];
                }
            }

            return NdArray.fromArray(output);
        }
        @Override
        protected NdArray needPreviousBackward(NdArray i_gy, NdArray i_prevInput)
        {
            double[] gxData = new double[this.inputCount];

            for (int i = 0; i < i_gy.length(); i++)
            {
                int indexOffset = this.inputCount * i;
                double gyData = i_gy.data[i];

                for (int j = 0; j < this.inputCount; j++)
                {
                    this.gW.data[indexOffset + j] += i_prevInput.data[j] * gyData;

                    gxData[j] += this.W.data[indexOffset + j] * gyData;
                }

                this.gb.data[i] += gyData;
            }

            return new NdArray(gxData, i_prevInput.shape);
        }
		@Override
		public Object deepCopy() {
			return new Linear(this);
		}
    }

