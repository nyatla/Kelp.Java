package jp.nyatla.kelpjava.functions.connections;

import jp.nyatla.kelpjava.FunctionParameter;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.NeedPreviousInputFunction;

//    [Serializable]
    final public class Linear extends NeedPreviousInputFunction
    {
		private static final long serialVersionUID = 5124105736160517816L;
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
        	this(
        		i_inputCount,i_outputCount,
        		false,
        		initWeight(NdArray.zeros(i_outputCount,i_inputCount)),
        		NdArray.zeros(i_outputCount),
        		i_name);
        }
        
        public Linear(int i_inputCount, int i_outputCount, boolean i_noBias,NdArray i_initialW, NdArray i_initialb, String i_name)
        {
        	super(i_name, i_inputCount, i_outputCount);
            this.W = i_initialW;
            this.gW = NdArray.zerosLike(this.W);
            this.parameters = new FunctionParameter[i_noBias ? 1 : 2];
            this.parameters[0] = new FunctionParameter(this.W, this.gW, this.name + " W");
            //noBias=trueでもbiasを用意して更新しない
            this.b = i_initialb;
            this.gb = NdArray.zerosLike(this.b);
            if (!i_noBias)
            {
                this.parameters[1] = new FunctionParameter(this.b, this.gb, this.name + " b");
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

            return new NdArray(gxData, i_prevInput.shape.clone(),false);
        }
		@Override
		public Object deepCopy() {
			return new Linear(this);
		}
    }

