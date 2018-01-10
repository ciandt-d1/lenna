#!/usr/bin/env bash
# Bash3 Boilerplate. Copyright (c) 2014, kvz.io

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

SEMVER="${1:-}"

init_py=tf_image_classification/__init__.py
sed_param="s/\${VERSION}/${SEMVER}/"

sed -e "${sed_param}" -i ${init_py}
git add ${init_py}
git commit -m "Releasing %{SEMVER}"
git tag ${SEMVER}
git push origin --tags
