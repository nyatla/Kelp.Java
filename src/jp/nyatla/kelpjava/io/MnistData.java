package jp.nyatla.kelpjava.io;

public class MnistData
    {

        private double[][][][] X;
        private int[][] Tx;

        private double[][][][] Y;
        private int[][] Ty;

        public MnistData(MnistLabelFile i_trainlabel,MnistImageFile i_traindata,MnistLabelFile i_teachlabel,MnistImageFile i_teachdata)
        {
            //トレーニングデータ
            this.X = new double[i_traindata.numberOfImages][1][28][28];
            //トレーニングデータラベル
            this.Tx = new int[i_trainlabel.numberOfItems][1];

            for (int i = 0; i < i_traindata.numberOfImages; i++)
            {
            	for(int p=0;p<28*28;p++){
            		this.X[i][0][p/28][p%28]=i_traindata.bitmapList[i][p];
            	}
                this.Tx[i][0] = (int)i_trainlabel.labelList[i];
            }

            //教師データ
            this.Y = new double[i_teachdata.numberOfImages][1][28][28];
            //教師データラベル
            this.Ty = new int[this.mnistDataLoader.TeachData.Length][];
            
            for (int i = 0; i < this.mnistDataLoader.TeachData.Length; i++)
            {
                Buffer.BlockCopy(this.mnistDataLoader.TeachData[i].Select(val => val / 255.0).ToArray(), 0, this.Y[i], 0, sizeof(double) *this.Y[i].Length);
                this.Ty[i] = new[] { (int)this.mnistDataLoader.TeachLabel[i] };
            }
        }

        public MnistDataSet GetRandomYSet(int dataCount)
        {
            List<double[,,]> listY = new List<double[,,]>();
            List<int[]> listTy = new List<int[]>();

            for (int j = 0; j < dataCount; j++)
            {
                int index = Mother.Dice.Next(this.Y.Length);

                listY.Add(this.Y[index]);
                listTy.Add(this.Ty[index]);
            }

            return new MnistDataSet(listY.ToArray(),listTy.ToArray());
        }

        public MnistDataSet GetRandomXSet(int dataCount)
        {
            List<double[,,]> listX = new List<double[,,]>();
            List<int[]> listTx = new List<int[]>();

            for (int j = 0; j < dataCount; j++)
            {
                int index = Mother.Dice.Next(this.X.Length);

                listX.Add(this.X[index]);
                listTx.Add(this.Tx[index]);
            }

            return new MnistDataSet(listX.ToArray(), listTx.ToArray());
        }
    }



