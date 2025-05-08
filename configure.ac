AC_INIT([bookish-octo-engine], [1.0], [info@example.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([
 Makefile
 src/Makefile
])

# Check for libnoodlenet
AC_CHECK_LIB([noodlenet], [noodlenet_load], [], [
  AC_MSG_ERROR([libnoodlenet is required but not found])
])
AC_CHECK_HEADERS([noodlenet.h], [], [
  AC_MSG_ERROR([noodlenet.h is required but not found])
])

AC_OUTPUT
