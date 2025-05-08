AC_INIT([bookish-octo-engine], [1.0], [info@example.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([
 Makefile
 src/Makefile
])

# Check for pkg-config
PKG_PROG_PKG_CONFIG

# Check for libnoodlenet using pkg-config
PKG_CHECK_MODULES([NOODLENET], [libnoodlenet >= 1.0], [], [
  AC_MSG_ERROR([libnoodlenet >= 1.0 is required but not found.
  Please make sure libnoodlenet is installed and pkg-config can find it.])
])

AC_OUTPUT
