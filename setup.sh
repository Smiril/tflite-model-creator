#!/usr/bin/env bash
cd "$(dirname $0)"
set -e

mkdir -p .dfl
mkdir -p image
mkdir -p image/train
mkdir -p image/validate

is_arm64() {
  [ "$(uname -sm)" == "Darwin arm64" ]
}

is_arm64 && echo "Running on Apple M1/M2 chip"

echo -n "path/to/your/python3 : "
read pythonX

if [ ! -d .dfl/env ]; then
  virtualenv -p $pythonX .dfl/env
fi

source .dfl/env/bin/activate

$pythonX -m pip install --upgrade pip

version=$($pythonX -V | cut -f 2 -d ' ' | cut -f 1,2 -d .)
reqs_file='requirements.txt'

version_suffix=''
if [[ ! -z "$version" && -f "requirements_$version.txt" ]]; then
  version_suffix="_$version"
fi

architecture_suffix=''
if is_arm64 && [ -f "requirements${version_suffix}_arm64.txt" ]; then
  architecture_suffix="_arm64"
fi

reqs_file="requirements${version_suffix}${architecture_suffix}.txt"

echo "Using $reqs_file for $(python3 -V)"

pip --no-cache-dir install -r $reqs_file

echo "Done."
