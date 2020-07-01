# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# -----------------------------------------------------------------------------
	
if(EXISTS "${REPOSITORY_DIR}/build/ThirdParty/share/sqlite3.tar.bz2")
    set(URL "${REPOSITORY_DIR}/build/ThirdParty/share/sqlite3.tar.bz2")
else()
	set(URL "https://www.sqlite.org/2020/sqlite-autoconf-3320300.tar.gz")
endif()

message(STATUS "obtaining SQLITE3")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ sqlite3
	PREFIX ${EP_BASE}/sqlite3
	URL ${URL}
	UPDATE_DISCONNECTED 1
	QUIET
	)
	
# No build. This is a header only package

FILE(APPEND "${EXPORT_FILE_NAME}" "sqlite3_INCLUDE_DIRS@@@${sqlite3_SOURCE_DIR}\n")
