###
### Benchmark script for Tacotron2
###
### For inference scenario, it is recommended to use single socket (or single NUMA node)
### since PyTorch is dynamic graph and will trigger immediate buffer allocation per
### each instance
###
### 1. customize 3rd party memory allocator (you can use jemalloc or tcmalloc)
###  e.g. jemalloc: https://github.com/jemalloc/jemalloc/wiki/Getting-Started
###     a) download from release: https://github.com/jemalloc/jemalloc/releases
###     b) tar -jxvf jemalloc-5.2.0.tar.bz2
###     c) ./configure
###        make
###     d) cd /home/mingfeim/packages/jemalloc-5.2.0/bin
###        chmod 777 jemalloc-config
###     e) LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so your_script.sh
###
### 2. regulate OpenMP threads: OMP_NUM_THREADS, KMP_AFFINITY, numactl as following
###
###
### Steps to run:
### 1. download pretrained model from https://github.com/mingfeima/tacotron2#training-using-a-pre-trained-model
###    move to models/tacotron2_statedict.pt
### 2. ./run_inference_cpu.sh
###    ./run_inference_cpu.sh --quantize
###

ARGS=""
if [[ "$1" == "--quantize" ]]; then
  ARGS="$ARGS --quantize"
  echo "### dynamic quantization"
  shift
fi
if [[ "$1" == "--profile" ]]; then
  ARGS="$ARGS --profile"
  echo "### start autograd profiler"
  shift
fi

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so

CORES=`lscpu | grep Core | awk '{print $4}'`
CORES=8
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

export $KMP_SETTING
echo -e "\n### using $KMP_SETTING"

export OMP_NUM_THREADS=$CORES
echo -e "### using OMP_NUM_THREADS=$CORES"
echo -e "### using $PREFIX\n"

$PREFIX python inference.py $ARGS
