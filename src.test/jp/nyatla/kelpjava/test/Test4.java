package jp.nyatla.kelpjava.test;

import java.io.File;
import java.io.IOException;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Optimizer;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.functions.activations.Sigmoid;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.io.MnistData;
import jp.nyatla.kelpjava.io.MnistData.DataSet;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.MomentumSGD;

    //MLPによるMNIST（手書き文字）の学習
    class Test4
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        final static int BATCH_DATA_COUNT = 20;

        //一世代あたりの訓練回数
        final static int TRAIN_DATA_COUNT = 3000; // = 60000 / 20

        //性能評価時のデータ数
        final static int TEST_DATA_COUNT = 200;


        public static void main(String[] args) throws IOException
        {
            //MNISTのデータを用意する
            System.out.println("MNIST Data Loading...");
            MnistData train = new MnistData(new File("data/train-images.idx3-ubyte"),new File("data/train-labels.idx1-ubyte"));
            MnistData teach = new MnistData(new File("data/t10k-images.idx3-ubyte"),new File("data/t10k-labels.idx1-ubyte"));


            System.out.println("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, 1024, "l1 Linear"),
                new Sigmoid("l1 Sigmoid"),
                new Linear(1024, 10, "l2 Linear")
            );
            //optimizerを宣言
            nn.setOptimizer(new Optimizer[]{new MomentumSGD()});

            //三世代学習
            for (int epoch = 0; epoch < 3; epoch++)
            {
                System.out.println("epoch " + (epoch + 1));

                //全体での誤差を集計
                double[] totalLoss=new double[TRAIN_DATA_COUNT];
                Trainer trainer=new Trainer();
                LossFunction loss=new SoftmaxCrossEntropy();
                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT+1; i++)
                {
                    //訓練データからランダムにデータを取得
                	DataSet datasetX=train.getRandomDataSet(20);

                    //バッチ学習を並列実行する
                    double sumLoss = trainer.train(nn,datasetX.image,datasetX.label,loss);
                    totalLoss[i-1]=sumLoss;

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        System.out.println("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        //結果出力
                        System.out.println("total loss " + JavaUtils.average(totalLoss));
                        System.out.println("local loss " + sumLoss);

                        System.out.println("\nTesting...");
                        
                        //テストデータからランダムにデータを取得
                        DataSet datasetY = teach.getRandomDataSet(TEST_DATA_COUNT);

                        //テストを実行
                        double accuracy = trainer.accuracy(nn, datasetY.image, datasetY.label);
                        System.out.println("accuracy " + accuracy);
                    }
                }
            }
        }
    }

