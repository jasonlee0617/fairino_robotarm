#!/usr/bin/env bash
# Source this file to use this package's pinned acados and CasADi toolchain.

MPC_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPC_PACKAGE_DIR="$(cd "${MPC_SCRIPTS_DIR}/.." && pwd)"
MPC_TOOLBOX_DIR="$(cd "${MPC_PACKAGE_DIR}/../mpc_toolbox" && pwd)"

if ! python3 -c 'import deprecated' >/dev/null 2>&1; then
    echo "[ERROR] Missing Python module 'deprecated' required by acados_template." >&2
    echo "Install it on Ubuntu with: sudo apt install -y python3-deprecated" >&2
    return 1 2>/dev/null || exit 1
fi

export ACADOS_INSTALL_DIR="${MPC_TOOLBOX_DIR}/acados"
export ACADOS_SOURCE_DIR="${ACADOS_INSTALL_DIR}"
export CASADI_DIR="${MPC_TOOLBOX_DIR}/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export LD_LIBRARY_PATH="${ACADOS_INSTALL_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${CASADI_DIR}:${ACADOS_INSTALL_DIR}/interfaces/acados_template${PYTHONPATH:+:${PYTHONPATH}}"
echo "[OK] local acados env: ACADOS_INSTALL_DIR=${ACADOS_INSTALL_DIR} CASADI_DIR=${CASADI_DIR}"
