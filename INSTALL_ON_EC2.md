# Install These Updates on EC2

Copy the files in this zip into your repo or directly onto EC2.

## EC2 pipeline path used during the live run

```bash
mkdir -p /home/ubuntu/pipeline
cp pipeline/*.py /home/ubuntu/pipeline/
cp scripts/run_ec2_pipeline.sh /home/ubuntu/run_ec2_pipeline.sh
chmod +x /home/ubuntu/run_ec2_pipeline.sh
```

## Spark 3.4.1 compatible Delta setup

```bash
SPARK_HOME=$(python3 -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))")

python3 -m pip uninstall -y delta-spark
python3 -m pip install "delta-spark==2.4.0"

cd $SPARK_HOME/jars
rm -f delta-*.jar
wget -q https://repo1.maven.org/maven2/io/delta/delta-core_2.12/2.4.0/delta-core_2.12-2.4.0.jar
wget -q https://repo1.maven.org/maven2/io/delta/delta-storage/2.4.0/delta-storage-2.4.0.jar
wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.4_2.12/1.5.2/iceberg-spark-runtime-3.4_2.12-1.5.2.jar
```

## Spark defaults

Do not put secrets in `spark-defaults.conf`.

```bash
SPARK_HOME=$(python3 -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))")
cp config/spark-defaults.spark34.conf $SPARK_HOME/conf/spark-defaults.conf
aws configure
```

## Run safely

```bash
screen -S pipeline
/home/ubuntu/run_ec2_pipeline.sh
```

Detach from screen with `Ctrl+A`, then `D`.
