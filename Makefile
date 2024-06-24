########################################################################################
#
# DO NOT MODIFY!!!
# If necessary, override the corresponding variable and/or target, or create new ones
# in one of the following files, depending on the nature of the override :
#
# `Makefile.variables`, `Makefile.targets` or `Makefile.private`,
#
# The only valid reason to modify this file is to fix a bug or to add new
# files to include.
########################################################################################

#Include base makefile
include .make/base.make
# Include custom targets and variables
-include Makefile.targets
-include Makefile.variables
# Private variables and targets import to override variables for local
-include Makefile.private
