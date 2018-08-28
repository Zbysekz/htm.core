# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_engine_internal', [dirname(__file__)])
        except ImportError:
            import _engine_internal
            return _engine_internal
        if fp is not None:
            try:
                _mod = imp.load_module('_engine_internal', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _engine_internal = swig_import_helper()
    del swig_import_helper
else:
    import _engine_internal
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


def _swig_setattr_nondynamic_method(set):
    def set_attr(self,name,value):
        if (name == "thisown"): return self.this.own(value)
        if hasattr(self,name) or (name == "this"):
            set(self,name,value)
        else:
            raise AttributeError("You cannot add attributes to %s" % self)
    return set_attr


# Support iteration of swig-generated collections in Python.

class IterableCollection(object):
  def __init__(self, collection):
    self._position = 0
    self._collection = collection

  def next(self):
    if self._position == self._collection.getCount():
      raise StopIteration

    val = self._collection.getByIndex(self._position)
    self._position += 1
    return val



class IterablePair(object):
  def __init__(self, pair):
    self._position = 0
    self._pair = pair

  def next(self):
    if self._position == 2:
      raise StopIteration

    val = getattr(self._pair, "first" if self._position == 0 else "second")
    self._position += 1
    return val


class SwigPyIterator(object):
    """Proxy of C++ swig::SwigPyIterator class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _engine_internal.delete_SwigPyIterator
    def value(self):
        """value(self) -> PyObject *"""
        return _engine_internal.SwigPyIterator_value(self)

    def incr(self, n=1):
        """incr(self, n=1) -> SwigPyIterator"""
        return _engine_internal.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        """decr(self, n=1) -> SwigPyIterator"""
        return _engine_internal.SwigPyIterator_decr(self, n)

    def distance(self, *args, **kwargs):
        """distance(self, x) -> ptrdiff_t"""
        return _engine_internal.SwigPyIterator_distance(self, *args, **kwargs)

    def equal(self, *args, **kwargs):
        """equal(self, x) -> bool"""
        return _engine_internal.SwigPyIterator_equal(self, *args, **kwargs)

    def copy(self):
        """copy(self) -> SwigPyIterator"""
        return _engine_internal.SwigPyIterator_copy(self)

    def next(self):
        """next(self) -> PyObject *"""
        return _engine_internal.SwigPyIterator_next(self)

    def __next__(self):
        """__next__(self) -> PyObject *"""
        return _engine_internal.SwigPyIterator___next__(self)

    def previous(self):
        """previous(self) -> PyObject *"""
        return _engine_internal.SwigPyIterator_previous(self)

    def advance(self, *args, **kwargs):
        """advance(self, n) -> SwigPyIterator"""
        return _engine_internal.SwigPyIterator_advance(self, *args, **kwargs)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, x) -> bool"""
        return _engine_internal.SwigPyIterator___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, x) -> bool"""
        return _engine_internal.SwigPyIterator___ne__(self, *args, **kwargs)

    def __iadd__(self, *args, **kwargs):
        """__iadd__(self, n) -> SwigPyIterator"""
        return _engine_internal.SwigPyIterator___iadd__(self, *args, **kwargs)

    def __isub__(self, *args, **kwargs):
        """__isub__(self, n) -> SwigPyIterator"""
        return _engine_internal.SwigPyIterator___isub__(self, *args, **kwargs)

    def __add__(self, *args, **kwargs):
        """__add__(self, n) -> SwigPyIterator"""
        return _engine_internal.SwigPyIterator___add__(self, *args, **kwargs)

    def __sub__(self, *args):
        """
        __sub__(self, n) -> SwigPyIterator
        __sub__(self, x) -> ptrdiff_t
        """
        return _engine_internal.SwigPyIterator___sub__(self, *args)

    def __iter__(self): return self
SwigPyIterator_swigregister = _engine_internal.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class StringVec(object):
    """Proxy of C++ std::vector<(std::string)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(self):
        """iterator(self) -> SwigPyIterator"""
        return _engine_internal.StringVec_iterator(self)

    def __iter__(self): return self.iterator()
    def __nonzero__(self):
        """__nonzero__(self) -> bool"""
        return _engine_internal.StringVec___nonzero__(self)

    def __bool__(self):
        """__bool__(self) -> bool"""
        return _engine_internal.StringVec___bool__(self)

    def __len__(self):
        """__len__(self) -> std::vector< std::string >::size_type"""
        return _engine_internal.StringVec___len__(self)

    def pop(self):
        """pop(self) -> std::vector< std::string >::value_type"""
        return _engine_internal.StringVec_pop(self)

    def __getslice__(self, *args, **kwargs):
        """__getslice__(self, i, j) -> StringVec"""
        return _engine_internal.StringVec___getslice__(self, *args, **kwargs)

    def __setslice__(self, *args, **kwargs):
        """__setslice__(self, i, j, v=std::vector< std::string,std::allocator< std::string > >())"""
        return _engine_internal.StringVec___setslice__(self, *args, **kwargs)

    def __delslice__(self, *args, **kwargs):
        """__delslice__(self, i, j)"""
        return _engine_internal.StringVec___delslice__(self, *args, **kwargs)

    def __delitem__(self, *args):
        """
        __delitem__(self, i)
        __delitem__(self, slice)
        """
        return _engine_internal.StringVec___delitem__(self, *args)

    def __getitem__(self, *args):
        """
        __getitem__(self, slice) -> StringVec
        __getitem__(self, i) -> std::vector< std::string >::value_type const &
        """
        return _engine_internal.StringVec___getitem__(self, *args)

    def __setitem__(self, *args):
        """
        __setitem__(self, slice, v)
        __setitem__(self, slice)
        __setitem__(self, i, x)
        """
        return _engine_internal.StringVec___setitem__(self, *args)

    def append(self, *args, **kwargs):
        """append(self, x)"""
        return _engine_internal.StringVec_append(self, *args, **kwargs)

    def empty(self):
        """empty(self) -> bool"""
        return _engine_internal.StringVec_empty(self)

    def size(self):
        """size(self) -> std::vector< std::string >::size_type"""
        return _engine_internal.StringVec_size(self)

    def clear(self):
        """clear(self)"""
        return _engine_internal.StringVec_clear(self)

    def swap(self, *args, **kwargs):
        """swap(self, v)"""
        return _engine_internal.StringVec_swap(self, *args, **kwargs)

    def get_allocator(self):
        """get_allocator(self) -> std::vector< std::string >::allocator_type"""
        return _engine_internal.StringVec_get_allocator(self)

    def begin(self):
        """begin(self) -> std::vector< std::string >::iterator"""
        return _engine_internal.StringVec_begin(self)

    def end(self):
        """end(self) -> std::vector< std::string >::iterator"""
        return _engine_internal.StringVec_end(self)

    def rbegin(self):
        """rbegin(self) -> std::vector< std::string >::reverse_iterator"""
        return _engine_internal.StringVec_rbegin(self)

    def rend(self):
        """rend(self) -> std::vector< std::string >::reverse_iterator"""
        return _engine_internal.StringVec_rend(self)

    def pop_back(self):
        """pop_back(self)"""
        return _engine_internal.StringVec_pop_back(self)

    def erase(self, *args):
        """
        erase(self, pos) -> std::vector< std::string >::iterator
        erase(self, first, last) -> std::vector< std::string >::iterator
        """
        return _engine_internal.StringVec_erase(self, *args)

    def __init__(self, *args): 
        """
        __init__(self) -> StringVec
        __init__(self, arg2) -> StringVec
        __init__(self, size) -> StringVec
        __init__(self, size, value) -> StringVec
        """
        this = _engine_internal.new_StringVec(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args, **kwargs):
        """push_back(self, x)"""
        return _engine_internal.StringVec_push_back(self, *args, **kwargs)

    def front(self):
        """front(self) -> std::vector< std::string >::value_type const &"""
        return _engine_internal.StringVec_front(self)

    def back(self):
        """back(self) -> std::vector< std::string >::value_type const &"""
        return _engine_internal.StringVec_back(self)

    def assign(self, *args, **kwargs):
        """assign(self, n, x)"""
        return _engine_internal.StringVec_assign(self, *args, **kwargs)

    def resize(self, *args):
        """
        resize(self, new_size)
        resize(self, new_size, x)
        """
        return _engine_internal.StringVec_resize(self, *args)

    def insert(self, *args):
        """
        insert(self, pos, x) -> std::vector< std::string >::iterator
        insert(self, pos, n, x)
        """
        return _engine_internal.StringVec_insert(self, *args)

    def reserve(self, *args, **kwargs):
        """reserve(self, n)"""
        return _engine_internal.StringVec_reserve(self, *args, **kwargs)

    def capacity(self):
        """capacity(self) -> std::vector< std::string >::size_type"""
        return _engine_internal.StringVec_capacity(self)

    __swig_destroy__ = _engine_internal.delete_StringVec
StringVec_swigregister = _engine_internal.StringVec_swigregister
StringVec_swigregister(StringVec)

NTA_BasicType_Byte = _engine_internal.NTA_BasicType_Byte
NTA_BasicType_Int16 = _engine_internal.NTA_BasicType_Int16
NTA_BasicType_UInt16 = _engine_internal.NTA_BasicType_UInt16
NTA_BasicType_Int32 = _engine_internal.NTA_BasicType_Int32
NTA_BasicType_UInt32 = _engine_internal.NTA_BasicType_UInt32
NTA_BasicType_Int64 = _engine_internal.NTA_BasicType_Int64
NTA_BasicType_UInt64 = _engine_internal.NTA_BasicType_UInt64
NTA_BasicType_Real32 = _engine_internal.NTA_BasicType_Real32
NTA_BasicType_Real64 = _engine_internal.NTA_BasicType_Real64
NTA_BasicType_Handle = _engine_internal.NTA_BasicType_Handle
NTA_BasicType_Bool = _engine_internal.NTA_BasicType_Bool
NTA_BasicType_Last = _engine_internal.NTA_BasicType_Last
NTA_BasicType_Real = _engine_internal.NTA_BasicType_Real
NTA_REAL_TYPE_STRING = _engine_internal.NTA_REAL_TYPE_STRING
NTA_LogLevel_None = _engine_internal.NTA_LogLevel_None
NTA_LogLevel_Minimal = _engine_internal.NTA_LogLevel_Minimal
NTA_LogLevel_Normal = _engine_internal.NTA_LogLevel_Normal
NTA_LogLevel_Verbose = _engine_internal.NTA_LogLevel_Verbose
LogLevel_None = _engine_internal.LogLevel_None
LogLevel_Minimal = _engine_internal.LogLevel_Minimal
LogLevel_Normal = _engine_internal.LogLevel_Normal
LogLevel_Verbose = _engine_internal.LogLevel_Verbose
class BasicType(object):
    """Proxy of C++ nupic::BasicType class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    def isValid(*args, **kwargs):
        """isValid(t) -> bool"""
        return _engine_internal.BasicType_isValid(*args, **kwargs)

    isValid = staticmethod(isValid)
    def getName(*args, **kwargs):
        """getName(t) -> char const *"""
        return _engine_internal.BasicType_getName(*args, **kwargs)

    getName = staticmethod(getName)
    def getSize(*args, **kwargs):
        """getSize(t) -> size_t"""
        return _engine_internal.BasicType_getSize(*args, **kwargs)

    getSize = staticmethod(getSize)
    def parse(*args, **kwargs):
        """parse(s) -> NTA_BasicType"""
        return _engine_internal.BasicType_parse(*args, **kwargs)

    parse = staticmethod(parse)
    def convertArray(*args, **kwargs):
        """convertArray(toPtr, toType, fromPtr, fromType, count)"""
        return _engine_internal.BasicType_convertArray(*args, **kwargs)

    convertArray = staticmethod(convertArray)
    __swig_destroy__ = _engine_internal.delete_BasicType
BasicType_swigregister = _engine_internal.BasicType_swigregister
BasicType_swigregister(BasicType)

def BasicType_isValid(*args, **kwargs):
  """BasicType_isValid(t) -> bool"""
  return _engine_internal.BasicType_isValid(*args, **kwargs)

def BasicType_getName(*args, **kwargs):
  """BasicType_getName(t) -> char const *"""
  return _engine_internal.BasicType_getName(*args, **kwargs)

def BasicType_getSize(*args, **kwargs):
  """BasicType_getSize(t) -> size_t"""
  return _engine_internal.BasicType_getSize(*args, **kwargs)

def BasicType_parse(*args, **kwargs):
  """BasicType_parse(s) -> NTA_BasicType"""
  return _engine_internal.BasicType_parse(*args, **kwargs)

def BasicType_convertArray(*args, **kwargs):
  """BasicType_convertArray(toPtr, toType, fromPtr, fromType, count)"""
  return _engine_internal.BasicType_convertArray(*args, **kwargs)

class Exception(object):
    """Proxy of C++ nupic::Exception class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        """__init__(self, filename, lineno, message, stacktrace="") -> Exception"""
        this = _engine_internal.new_Exception(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_Exception
    def what(self):
        """what(self) -> char const *"""
        return _engine_internal.Exception_what(self)

    def getFilename(self):
        """getFilename(self) -> char const *"""
        return _engine_internal.Exception_getFilename(self)

    def getLineNumber(self):
        """getLineNumber(self) -> nupic::UInt32"""
        return _engine_internal.Exception_getLineNumber(self)

    def getMessage(self):
        """getMessage(self) -> char const *"""
        return _engine_internal.Exception_getMessage(self)

    def getStackTrace(self):
        """getStackTrace(self) -> char const *"""
        return _engine_internal.Exception_getStackTrace(self)

Exception_swigregister = _engine_internal.Exception_swigregister
Exception_swigregister(Exception)

class UInt32Set(object):
    """Proxy of C++ std::set<(nupic::UInt32)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(self):
        """iterator(self) -> SwigPyIterator"""
        return _engine_internal.UInt32Set_iterator(self)

    def __iter__(self): return self.iterator()
    def __nonzero__(self):
        """__nonzero__(self) -> bool"""
        return _engine_internal.UInt32Set___nonzero__(self)

    def __bool__(self):
        """__bool__(self) -> bool"""
        return _engine_internal.UInt32Set___bool__(self)

    def __len__(self):
        """__len__(self) -> std::set< unsigned int >::size_type"""
        return _engine_internal.UInt32Set___len__(self)

    def append(self, *args, **kwargs):
        """append(self, x)"""
        return _engine_internal.UInt32Set_append(self, *args, **kwargs)

    def __contains__(self, *args, **kwargs):
        """__contains__(self, x) -> bool"""
        return _engine_internal.UInt32Set___contains__(self, *args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> std::set< unsigned int >::value_type"""
        return _engine_internal.UInt32Set___getitem__(self, *args, **kwargs)

    def add(self, *args, **kwargs):
        """add(self, x)"""
        return _engine_internal.UInt32Set_add(self, *args, **kwargs)

    def discard(self, *args, **kwargs):
        """discard(self, x)"""
        return _engine_internal.UInt32Set_discard(self, *args, **kwargs)

    def __init__(self, *args): 
        """
        __init__(self, arg2) -> UInt32Set
        __init__(self) -> UInt32Set
        __init__(self, arg2) -> UInt32Set
        """
        this = _engine_internal.new_UInt32Set(*args)
        try: self.this.append(this)
        except: self.this = this
    def empty(self):
        """empty(self) -> bool"""
        return _engine_internal.UInt32Set_empty(self)

    def size(self):
        """size(self) -> std::set< unsigned int >::size_type"""
        return _engine_internal.UInt32Set_size(self)

    def clear(self):
        """clear(self)"""
        return _engine_internal.UInt32Set_clear(self)

    def swap(self, *args, **kwargs):
        """swap(self, v)"""
        return _engine_internal.UInt32Set_swap(self, *args, **kwargs)

    def count(self, *args, **kwargs):
        """count(self, x) -> std::set< unsigned int >::size_type"""
        return _engine_internal.UInt32Set_count(self, *args, **kwargs)

    def begin(self):
        """begin(self) -> std::set< unsigned int >::iterator"""
        return _engine_internal.UInt32Set_begin(self)

    def end(self):
        """end(self) -> std::set< unsigned int >::iterator"""
        return _engine_internal.UInt32Set_end(self)

    def rbegin(self):
        """rbegin(self) -> std::set< unsigned int >::reverse_iterator"""
        return _engine_internal.UInt32Set_rbegin(self)

    def rend(self):
        """rend(self) -> std::set< unsigned int >::reverse_iterator"""
        return _engine_internal.UInt32Set_rend(self)

    def erase(self, *args):
        """
        erase(self, x) -> std::set< unsigned int >::size_type
        erase(self, pos)
        erase(self, first, last)
        """
        return _engine_internal.UInt32Set_erase(self, *args)

    def find(self, *args, **kwargs):
        """find(self, x) -> std::set< unsigned int >::iterator"""
        return _engine_internal.UInt32Set_find(self, *args, **kwargs)

    def lower_bound(self, *args, **kwargs):
        """lower_bound(self, x) -> std::set< unsigned int >::iterator"""
        return _engine_internal.UInt32Set_lower_bound(self, *args, **kwargs)

    def upper_bound(self, *args, **kwargs):
        """upper_bound(self, x) -> std::set< unsigned int >::iterator"""
        return _engine_internal.UInt32Set_upper_bound(self, *args, **kwargs)

    def equal_range(self, *args, **kwargs):
        """equal_range(self, x) -> std::pair< std::set< unsigned int >::iterator,std::set< unsigned int >::iterator >"""
        return _engine_internal.UInt32Set_equal_range(self, *args, **kwargs)

    def insert(self, *args, **kwargs):
        """insert(self, __x) -> std::pair< std::set< unsigned int >::iterator,bool >"""
        return _engine_internal.UInt32Set_insert(self, *args, **kwargs)

    __swig_destroy__ = _engine_internal.delete_UInt32Set
UInt32Set_swigregister = _engine_internal.UInt32Set_swigregister
UInt32Set_swigregister(UInt32Set)

class Dimset(object):
    """Proxy of C++ std::vector<(size_t)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(self):
        """iterator(self) -> SwigPyIterator"""
        return _engine_internal.Dimset_iterator(self)

    def __iter__(self): return self.iterator()
    def __nonzero__(self):
        """__nonzero__(self) -> bool"""
        return _engine_internal.Dimset___nonzero__(self)

    def __bool__(self):
        """__bool__(self) -> bool"""
        return _engine_internal.Dimset___bool__(self)

    def __len__(self):
        """__len__(self) -> std::vector< size_t >::size_type"""
        return _engine_internal.Dimset___len__(self)

    def pop(self):
        """pop(self) -> std::vector< size_t >::value_type"""
        return _engine_internal.Dimset_pop(self)

    def __getslice__(self, *args, **kwargs):
        """__getslice__(self, i, j) -> Dimset"""
        return _engine_internal.Dimset___getslice__(self, *args, **kwargs)

    def __setslice__(self, *args, **kwargs):
        """__setslice__(self, i, j, v=std::vector< size_t,std::allocator< size_t > >())"""
        return _engine_internal.Dimset___setslice__(self, *args, **kwargs)

    def __delslice__(self, *args, **kwargs):
        """__delslice__(self, i, j)"""
        return _engine_internal.Dimset___delslice__(self, *args, **kwargs)

    def __delitem__(self, *args):
        """
        __delitem__(self, i)
        __delitem__(self, slice)
        """
        return _engine_internal.Dimset___delitem__(self, *args)

    def __getitem__(self, *args):
        """
        __getitem__(self, slice) -> Dimset
        __getitem__(self, i) -> std::vector< size_t >::value_type const &
        """
        return _engine_internal.Dimset___getitem__(self, *args)

    def __setitem__(self, *args):
        """
        __setitem__(self, slice, v)
        __setitem__(self, slice)
        __setitem__(self, i, x)
        """
        return _engine_internal.Dimset___setitem__(self, *args)

    def append(self, *args, **kwargs):
        """append(self, x)"""
        return _engine_internal.Dimset_append(self, *args, **kwargs)

    def empty(self):
        """empty(self) -> bool"""
        return _engine_internal.Dimset_empty(self)

    def size(self):
        """size(self) -> std::vector< size_t >::size_type"""
        return _engine_internal.Dimset_size(self)

    def clear(self):
        """clear(self)"""
        return _engine_internal.Dimset_clear(self)

    def swap(self, *args, **kwargs):
        """swap(self, v)"""
        return _engine_internal.Dimset_swap(self, *args, **kwargs)

    def get_allocator(self):
        """get_allocator(self) -> std::vector< size_t >::allocator_type"""
        return _engine_internal.Dimset_get_allocator(self)

    def begin(self):
        """begin(self) -> std::vector< size_t >::iterator"""
        return _engine_internal.Dimset_begin(self)

    def end(self):
        """end(self) -> std::vector< size_t >::iterator"""
        return _engine_internal.Dimset_end(self)

    def rbegin(self):
        """rbegin(self) -> std::vector< size_t >::reverse_iterator"""
        return _engine_internal.Dimset_rbegin(self)

    def rend(self):
        """rend(self) -> std::vector< size_t >::reverse_iterator"""
        return _engine_internal.Dimset_rend(self)

    def pop_back(self):
        """pop_back(self)"""
        return _engine_internal.Dimset_pop_back(self)

    def erase(self, *args):
        """
        erase(self, pos) -> std::vector< size_t >::iterator
        erase(self, first, last) -> std::vector< size_t >::iterator
        """
        return _engine_internal.Dimset_erase(self, *args)

    def __init__(self, *args): 
        """
        __init__(self) -> Dimset
        __init__(self, arg2) -> Dimset
        __init__(self, size) -> Dimset
        __init__(self, size, value) -> Dimset
        """
        this = _engine_internal.new_Dimset(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args, **kwargs):
        """push_back(self, x)"""
        return _engine_internal.Dimset_push_back(self, *args, **kwargs)

    def front(self):
        """front(self) -> std::vector< size_t >::value_type const &"""
        return _engine_internal.Dimset_front(self)

    def back(self):
        """back(self) -> std::vector< size_t >::value_type const &"""
        return _engine_internal.Dimset_back(self)

    def assign(self, *args, **kwargs):
        """assign(self, n, x)"""
        return _engine_internal.Dimset_assign(self, *args, **kwargs)

    def resize(self, *args):
        """
        resize(self, new_size)
        resize(self, new_size, x)
        """
        return _engine_internal.Dimset_resize(self, *args)

    def insert(self, *args):
        """
        insert(self, pos, x) -> std::vector< size_t >::iterator
        insert(self, pos, n, x)
        """
        return _engine_internal.Dimset_insert(self, *args)

    def reserve(self, *args, **kwargs):
        """reserve(self, n)"""
        return _engine_internal.Dimset_reserve(self, *args, **kwargs)

    def capacity(self):
        """capacity(self) -> std::vector< size_t >::size_type"""
        return _engine_internal.Dimset_capacity(self)

    __swig_destroy__ = _engine_internal.delete_Dimset
Dimset_swigregister = _engine_internal.Dimset_swigregister
Dimset_swigregister(Dimset)

class Dimensions(Dimset):
    """Proxy of C++ nupic::Dimensions class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> Dimensions
        __init__(self, v) -> Dimensions
        __init__(self, x) -> Dimensions
        __init__(self, x, y) -> Dimensions
        __init__(self, x, y, z) -> Dimensions
        """
        this = _engine_internal.new_Dimensions(*args)
        try: self.this.append(this)
        except: self.this = this
    def getCount(self):
        """getCount(self) -> size_t"""
        return _engine_internal.Dimensions_getCount(self)

    def getDimensionCount(self):
        """getDimensionCount(self) -> size_t"""
        return _engine_internal.Dimensions_getDimensionCount(self)

    def getDimension(self, *args, **kwargs):
        """getDimension(self, index) -> size_t"""
        return _engine_internal.Dimensions_getDimension(self, *args, **kwargs)

    def isUnspecified(self):
        """isUnspecified(self) -> bool"""
        return _engine_internal.Dimensions_isUnspecified(self)

    def isDontcare(self):
        """isDontcare(self) -> bool"""
        return _engine_internal.Dimensions_isDontcare(self)

    def isSpecified(self):
        """isSpecified(self) -> bool"""
        return _engine_internal.Dimensions_isSpecified(self)

    def isOnes(self):
        """isOnes(self) -> bool"""
        return _engine_internal.Dimensions_isOnes(self)

    def isValid(self):
        """isValid(self) -> bool"""
        return _engine_internal.Dimensions_isValid(self)

    def getIndex(self, *args, **kwargs):
        """getIndex(self, coordinate) -> size_t"""
        return _engine_internal.Dimensions_getIndex(self, *args, **kwargs)

    def getCoordinate(self, *args, **kwargs):
        """getCoordinate(self, index) -> Dimset"""
        return _engine_internal.Dimensions_getCoordinate(self, *args, **kwargs)

    def toString(self, humanReadable=True):
        """toString(self, humanReadable=True) -> std::string"""
        return _engine_internal.Dimensions_toString(self, humanReadable)

    def promote(self, *args, **kwargs):
        """promote(self, newDimensionality)"""
        return _engine_internal.Dimensions_promote(self, *args, **kwargs)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, dims2) -> bool"""
        return _engine_internal.Dimensions___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, dims2) -> bool"""
        return _engine_internal.Dimensions___ne__(self, *args, **kwargs)

    __swig_destroy__ = _engine_internal.delete_Dimensions
Dimensions_swigregister = _engine_internal.Dimensions_swigregister
Dimensions_swigregister(Dimensions)

class Array(object):
    """Proxy of C++ nupic::Array class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> Array
        __init__(self, type, buffer, count) -> Array
        __init__(self, type) -> Array
        """
        this = _engine_internal.new_Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def copy(self):
        """copy(self) -> Array"""
        return _engine_internal.Array_copy(self)

    def copyFrom(self, *args, **kwargs):
        """copyFrom(self, type, buf, size)"""
        return _engine_internal.Array_copyFrom(self, *args, **kwargs)

    def zeroCopy(self, *args, **kwargs):
        """zeroCopy(self, a)"""
        return _engine_internal.Array_zeroCopy(self, *args, **kwargs)

    def asVector(self):
        """asVector(self) -> std::vector< nupic::UInt32,std::allocator< nupic::UInt32 > >"""
        return _engine_internal.Array_asVector(self)

    def fromVector(self, *args, **kwargs):
        """fromVector(self, vect)"""
        return _engine_internal.Array_fromVector(self, *args, **kwargs)

    def convertInto(self, *args, **kwargs):
        """convertInto(self, a, offset=0)"""
        return _engine_internal.Array_convertInto(self, *args, **kwargs)

    def nonZero(self):
        """nonZero(self) -> Array"""
        return _engine_internal.Array_nonZero(self)

    def subset(self, *args, **kwargs):
        """subset(self, offset, count) -> Array"""
        return _engine_internal.Array_subset(self, *args, **kwargs)

    def invariant(self):
        """invariant(self)"""
        return _engine_internal.Array_invariant(self)

    __swig_destroy__ = _engine_internal.delete_Array
Array_swigregister = _engine_internal.Array_swigregister
Array_swigregister(Array)

class ArrayRef(object):
    """Proxy of C++ nupic::ArrayRef class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> ArrayRef
        __init__(self, type, buffer, count) -> ArrayRef
        __init__(self, type) -> ArrayRef
        __init__(self, other) -> ArrayRef
        """
        this = _engine_internal.new_ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getBuffer(self):
        """getBuffer(self) -> void const *"""
        return _engine_internal.ArrayRef_getBuffer(self)

    def invariant(self):
        """invariant(self)"""
        return _engine_internal.ArrayRef_invariant(self)

    __swig_destroy__ = _engine_internal.delete_ArrayRef
ArrayRef_swigregister = _engine_internal.ArrayRef_swigregister
ArrayRef_swigregister(ArrayRef)

class InputCollection(object):
    """Proxy of C++ nupic::Collection<(nupic::InputSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> InputCollection"""
        this = _engine_internal.new_InputCollection()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_InputCollection
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.InputCollection___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.InputCollection___ne__(self, *args, **kwargs)

    def getCount(self):
        """getCount(self) -> size_t"""
        return _engine_internal.InputCollection_getCount(self)

    def getByIndex(self, *args, **kwargs):
        """getByIndex(self, index) -> InputPair"""
        return _engine_internal.InputCollection_getByIndex(self, *args, **kwargs)

    def contains(self, *args, **kwargs):
        """contains(self, name) -> bool"""
        return _engine_internal.InputCollection_contains(self, *args, **kwargs)

    def getByName(self, *args, **kwargs):
        """getByName(self, name) -> InputSpec"""
        return _engine_internal.InputCollection_getByName(self, *args, **kwargs)

    def add(self, *args, **kwargs):
        """add(self, name, item)"""
        return _engine_internal.InputCollection_add(self, *args, **kwargs)

    def remove(self, *args, **kwargs):
        """remove(self, name)"""
        return _engine_internal.InputCollection_remove(self, *args, **kwargs)

InputCollection_swigregister = _engine_internal.InputCollection_swigregister
InputCollection_swigregister(InputCollection)

class OutputCollection(object):
    """Proxy of C++ nupic::Collection<(nupic::OutputSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> OutputCollection"""
        this = _engine_internal.new_OutputCollection()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_OutputCollection
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.OutputCollection___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.OutputCollection___ne__(self, *args, **kwargs)

    def getCount(self):
        """getCount(self) -> size_t"""
        return _engine_internal.OutputCollection_getCount(self)

    def getByIndex(self, *args, **kwargs):
        """getByIndex(self, index) -> OutputPair"""
        return _engine_internal.OutputCollection_getByIndex(self, *args, **kwargs)

    def contains(self, *args, **kwargs):
        """contains(self, name) -> bool"""
        return _engine_internal.OutputCollection_contains(self, *args, **kwargs)

    def getByName(self, *args, **kwargs):
        """getByName(self, name) -> OutputSpec"""
        return _engine_internal.OutputCollection_getByName(self, *args, **kwargs)

    def add(self, *args, **kwargs):
        """add(self, name, item)"""
        return _engine_internal.OutputCollection_add(self, *args, **kwargs)

    def remove(self, *args, **kwargs):
        """remove(self, name)"""
        return _engine_internal.OutputCollection_remove(self, *args, **kwargs)

OutputCollection_swigregister = _engine_internal.OutputCollection_swigregister
OutputCollection_swigregister(OutputCollection)

class ParameterCollection(object):
    """Proxy of C++ nupic::Collection<(nupic::ParameterSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> ParameterCollection"""
        this = _engine_internal.new_ParameterCollection()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_ParameterCollection
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.ParameterCollection___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.ParameterCollection___ne__(self, *args, **kwargs)

    def getCount(self):
        """getCount(self) -> size_t"""
        return _engine_internal.ParameterCollection_getCount(self)

    def getByIndex(self, *args, **kwargs):
        """getByIndex(self, index) -> ParameterPair"""
        return _engine_internal.ParameterCollection_getByIndex(self, *args, **kwargs)

    def contains(self, *args, **kwargs):
        """contains(self, name) -> bool"""
        return _engine_internal.ParameterCollection_contains(self, *args, **kwargs)

    def getByName(self, *args, **kwargs):
        """getByName(self, name) -> ParameterSpec"""
        return _engine_internal.ParameterCollection_getByName(self, *args, **kwargs)

    def add(self, *args, **kwargs):
        """add(self, name, item)"""
        return _engine_internal.ParameterCollection_add(self, *args, **kwargs)

    def remove(self, *args, **kwargs):
        """remove(self, name)"""
        return _engine_internal.ParameterCollection_remove(self, *args, **kwargs)

ParameterCollection_swigregister = _engine_internal.ParameterCollection_swigregister
ParameterCollection_swigregister(ParameterCollection)

class CommandCollection(object):
    """Proxy of C++ nupic::Collection<(nupic::CommandSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> CommandCollection"""
        this = _engine_internal.new_CommandCollection()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_CommandCollection
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.CommandCollection___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.CommandCollection___ne__(self, *args, **kwargs)

    def getCount(self):
        """getCount(self) -> size_t"""
        return _engine_internal.CommandCollection_getCount(self)

    def getByIndex(self, *args, **kwargs):
        """getByIndex(self, index) -> CommandPair"""
        return _engine_internal.CommandCollection_getByIndex(self, *args, **kwargs)

    def contains(self, *args, **kwargs):
        """contains(self, name) -> bool"""
        return _engine_internal.CommandCollection_contains(self, *args, **kwargs)

    def getByName(self, *args, **kwargs):
        """getByName(self, name) -> CommandSpec"""
        return _engine_internal.CommandCollection_getByName(self, *args, **kwargs)

    def add(self, *args, **kwargs):
        """add(self, name, item)"""
        return _engine_internal.CommandCollection_add(self, *args, **kwargs)

    def remove(self, *args, **kwargs):
        """remove(self, name)"""
        return _engine_internal.CommandCollection_remove(self, *args, **kwargs)

CommandCollection_swigregister = _engine_internal.CommandCollection_swigregister
CommandCollection_swigregister(CommandCollection)

class RegionCollection(object):
    """Proxy of C++ nupic::Collection<(p.nupic::Region)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> RegionCollection"""
        this = _engine_internal.new_RegionCollection()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_RegionCollection
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.RegionCollection___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.RegionCollection___ne__(self, *args, **kwargs)

    def getCount(self):
        """getCount(self) -> size_t"""
        return _engine_internal.RegionCollection_getCount(self)

    def getByIndex(self, *args, **kwargs):
        """getByIndex(self, index) -> RegionPair"""
        return _engine_internal.RegionCollection_getByIndex(self, *args, **kwargs)

    def contains(self, *args, **kwargs):
        """contains(self, name) -> bool"""
        return _engine_internal.RegionCollection_contains(self, *args, **kwargs)

    def getByName(self, *args, **kwargs):
        """getByName(self, name) -> Region"""
        return _engine_internal.RegionCollection_getByName(self, *args, **kwargs)

    def add(self, *args, **kwargs):
        """add(self, name, item)"""
        return _engine_internal.RegionCollection_add(self, *args, **kwargs)

    def remove(self, *args, **kwargs):
        """remove(self, name)"""
        return _engine_internal.RegionCollection_remove(self, *args, **kwargs)

RegionCollection_swigregister = _engine_internal.RegionCollection_swigregister
RegionCollection_swigregister(RegionCollection)

class LinkCollection(object):
    """Proxy of C++ nupic::Collection<(p.nupic::Link)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> LinkCollection"""
        this = _engine_internal.new_LinkCollection()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_LinkCollection
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.LinkCollection___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.LinkCollection___ne__(self, *args, **kwargs)

    def getCount(self):
        """getCount(self) -> size_t"""
        return _engine_internal.LinkCollection_getCount(self)

    def getByIndex(self, *args, **kwargs):
        """getByIndex(self, index) -> LinkPair"""
        return _engine_internal.LinkCollection_getByIndex(self, *args, **kwargs)

    def contains(self, *args, **kwargs):
        """contains(self, name) -> bool"""
        return _engine_internal.LinkCollection_contains(self, *args, **kwargs)

    def getByName(self, *args, **kwargs):
        """getByName(self, name) -> Link"""
        return _engine_internal.LinkCollection_getByName(self, *args, **kwargs)

    def add(self, *args, **kwargs):
        """add(self, name, item)"""
        return _engine_internal.LinkCollection_add(self, *args, **kwargs)

    def remove(self, *args, **kwargs):
        """remove(self, name)"""
        return _engine_internal.LinkCollection_remove(self, *args, **kwargs)

    def __iter__(self):
      return IterableCollection(self)

LinkCollection_swigregister = _engine_internal.LinkCollection_swigregister
LinkCollection_swigregister(LinkCollection)

class NuPIC(object):
    """Proxy of C++ nupic::NuPIC class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def init():
        """init()"""
        return _engine_internal.NuPIC_init()

    init = staticmethod(init)
    def shutdown():
        """shutdown()"""
        return _engine_internal.NuPIC_shutdown()

    shutdown = staticmethod(shutdown)
    def isInitialized():
        """isInitialized() -> bool"""
        return _engine_internal.NuPIC_isInitialized()

    isInitialized = staticmethod(isInitialized)
    def __init__(self): 
        """__init__(self) -> NuPIC"""
        this = _engine_internal.new_NuPIC()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_NuPIC
NuPIC_swigregister = _engine_internal.NuPIC_swigregister
NuPIC_swigregister(NuPIC)

def NuPIC_init():
  """NuPIC_init()"""
  return _engine_internal.NuPIC_init()

def NuPIC_shutdown():
  """NuPIC_shutdown()"""
  return _engine_internal.NuPIC_shutdown()

def NuPIC_isInitialized():
  """NuPIC_isInitialized() -> bool"""
  return _engine_internal.NuPIC_isInitialized()

class Network(object):
    """Proxy of C++ nupic::Network class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> Network"""
        this = _engine_internal.new_Network()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_Network
    def initialize(self):
        """initialize(self)"""
        return _engine_internal.Network_initialize(self)

    def save(self, *args, **kwargs):
        """save(self, f)"""
        return _engine_internal.Network_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, stream)"""
        return _engine_internal.Network_load(self, *args, **kwargs)

    def addRegion(self, *args):
        """
        addRegion(self, name, nodeType, nodeParams) -> Region
        addRegion(self, stream, name="") -> Region
        """
        return _engine_internal.Network_addRegion(self, *args)

    def removeRegion(self, *args, **kwargs):
        """removeRegion(self, name)"""
        return _engine_internal.Network_removeRegion(self, *args, **kwargs)

    def link(self, *args, **kwargs):
        """link(self, srcName, destName, linkType="", linkParams="", srcOutput="", destInput="", propagationDelay=0)"""
        return _engine_internal.Network_link(self, *args, **kwargs)

    def removeLink(self, *args, **kwargs):
        """removeLink(self, srcName, destName, srcOutputName="", destInputName="")"""
        return _engine_internal.Network_removeLink(self, *args, **kwargs)

    def getRegions(self):
        """getRegions(self) -> RegionCollection"""
        return _engine_internal.Network_getRegions(self)

    def getLinks(self):
        """getLinks(self) -> LinkCollection"""
        return _engine_internal.Network_getLinks(self)

    def setPhases(self, *args, **kwargs):
        """setPhases(self, name, phases)"""
        return _engine_internal.Network_setPhases(self, *args, **kwargs)

    def getPhases(self, *args, **kwargs):
        """getPhases(self, name) -> UInt32Set"""
        return _engine_internal.Network_getPhases(self, *args, **kwargs)

    def getMinPhase(self):
        """getMinPhase(self) -> nupic::UInt32"""
        return _engine_internal.Network_getMinPhase(self)

    def getMaxPhase(self):
        """getMaxPhase(self) -> nupic::UInt32"""
        return _engine_internal.Network_getMaxPhase(self)

    def setMinEnabledPhase(self, *args, **kwargs):
        """setMinEnabledPhase(self, minPhase)"""
        return _engine_internal.Network_setMinEnabledPhase(self, *args, **kwargs)

    def setMaxEnabledPhase(self, *args, **kwargs):
        """setMaxEnabledPhase(self, minPhase)"""
        return _engine_internal.Network_setMaxEnabledPhase(self, *args, **kwargs)

    def getMinEnabledPhase(self):
        """getMinEnabledPhase(self) -> nupic::UInt32"""
        return _engine_internal.Network_getMinEnabledPhase(self)

    def getMaxEnabledPhase(self):
        """getMaxEnabledPhase(self) -> nupic::UInt32"""
        return _engine_internal.Network_getMaxEnabledPhase(self)

    def run(self, *args, **kwargs):
        """run(self, n)"""
        return _engine_internal.Network_run(self, *args, **kwargs)

    def getCallbacks(self):
        """getCallbacks(self) -> nupic::Collection< nupic::Network::callbackItem > &"""
        return _engine_internal.Network_getCallbacks(self)

    def enableProfiling(self):
        """enableProfiling(self)"""
        return _engine_internal.Network_enableProfiling(self)

    def disableProfiling(self):
        """disableProfiling(self)"""
        return _engine_internal.Network_disableProfiling(self)

    def resetProfiling(self):
        """resetProfiling(self)"""
        return _engine_internal.Network_resetProfiling(self)

    def registerPyRegion(*args, **kwargs):
        """registerPyRegion(module, className)"""
        return _engine_internal.Network_registerPyRegion(*args, **kwargs)

    registerPyRegion = staticmethod(registerPyRegion)
    def registerCPPRegion(*args, **kwargs):
        """registerCPPRegion(name, wrapper)"""
        return _engine_internal.Network_registerCPPRegion(*args, **kwargs)

    registerCPPRegion = staticmethod(registerCPPRegion)
    def unregisterPyRegion(*args, **kwargs):
        """unregisterPyRegion(className)"""
        return _engine_internal.Network_unregisterPyRegion(*args, **kwargs)

    unregisterPyRegion = staticmethod(unregisterPyRegion)
    def unregisterCPPRegion(*args, **kwargs):
        """unregisterCPPRegion(name)"""
        return _engine_internal.Network_unregisterCPPRegion(*args, **kwargs)

    unregisterCPPRegion = staticmethod(unregisterCPPRegion)
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.Network___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.Network___ne__(self, *args, **kwargs)

Network_swigregister = _engine_internal.Network_swigregister
Network_swigregister(Network)

def Network_registerPyRegion(*args, **kwargs):
  """Network_registerPyRegion(module, className)"""
  return _engine_internal.Network_registerPyRegion(*args, **kwargs)

def Network_registerCPPRegion(*args, **kwargs):
  """Network_registerCPPRegion(name, wrapper)"""
  return _engine_internal.Network_registerCPPRegion(*args, **kwargs)

def Network_unregisterPyRegion(*args, **kwargs):
  """Network_unregisterPyRegion(className)"""
  return _engine_internal.Network_unregisterPyRegion(*args, **kwargs)

def Network_unregisterCPPRegion(*args, **kwargs):
  """Network_unregisterCPPRegion(name)"""
  return _engine_internal.Network_unregisterCPPRegion(*args, **kwargs)

class Region(object):
    """Proxy of C++ nupic::Region class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def getNetwork(self):
        """getNetwork(self) -> Network"""
        return _engine_internal.Region_getNetwork(self)

    def getName(self):
        """getName(self) -> std::string"""
        return _engine_internal.Region_getName(self)

    def getDimensions(self):
        """getDimensions(self) -> Dimensions"""
        return _engine_internal.Region_getDimensions(self)

    def setDimensions(self, *args, **kwargs):
        """setDimensions(self, dimensions)"""
        return _engine_internal.Region_setDimensions(self, *args, **kwargs)

    def getType(self):
        """getType(self) -> std::string"""
        return _engine_internal.Region_getType(self)

    def getSpec(self):
        """getSpec(self) -> Spec"""
        return _engine_internal.Region_getSpec(self)

    def getSpecFromType(*args, **kwargs):
        """getSpecFromType(nodeType) -> Spec"""
        return _engine_internal.Region_getSpecFromType(*args, **kwargs)

    getSpecFromType = staticmethod(getSpecFromType)
    def registerPyRegion(*args, **kwargs):
        """registerPyRegion(module, className)"""
        return _engine_internal.Region_registerPyRegion(*args, **kwargs)

    registerPyRegion = staticmethod(registerPyRegion)
    def registerCPPRegion(*args, **kwargs):
        """registerCPPRegion(name, wrapper)"""
        return _engine_internal.Region_registerCPPRegion(*args, **kwargs)

    registerCPPRegion = staticmethod(registerCPPRegion)
    def unregisterPyRegion(*args, **kwargs):
        """unregisterPyRegion(className)"""
        return _engine_internal.Region_unregisterPyRegion(*args, **kwargs)

    unregisterPyRegion = staticmethod(unregisterPyRegion)
    def unregisterCPPRegion(*args, **kwargs):
        """unregisterCPPRegion(name)"""
        return _engine_internal.Region_unregisterCPPRegion(*args, **kwargs)

    unregisterCPPRegion = staticmethod(unregisterCPPRegion)
    def getParameterInt32(self, *args, **kwargs):
        """getParameterInt32(self, name) -> nupic::Int32"""
        return _engine_internal.Region_getParameterInt32(self, *args, **kwargs)

    def getParameterUInt32(self, *args, **kwargs):
        """getParameterUInt32(self, name) -> nupic::UInt32"""
        return _engine_internal.Region_getParameterUInt32(self, *args, **kwargs)

    def getParameterInt64(self, *args, **kwargs):
        """getParameterInt64(self, name) -> nupic::Int64"""
        return _engine_internal.Region_getParameterInt64(self, *args, **kwargs)

    def getParameterUInt64(self, *args, **kwargs):
        """getParameterUInt64(self, name) -> nupic::UInt64"""
        return _engine_internal.Region_getParameterUInt64(self, *args, **kwargs)

    def getParameterReal32(self, *args, **kwargs):
        """getParameterReal32(self, name) -> nupic::Real32"""
        return _engine_internal.Region_getParameterReal32(self, *args, **kwargs)

    def getParameterReal64(self, *args, **kwargs):
        """getParameterReal64(self, name) -> nupic::Real64"""
        return _engine_internal.Region_getParameterReal64(self, *args, **kwargs)

    def getParameterHandle(self, *args, **kwargs):
        """getParameterHandle(self, name) -> nupic::Handle"""
        return _engine_internal.Region_getParameterHandle(self, *args, **kwargs)

    def getParameterBool(self, *args, **kwargs):
        """getParameterBool(self, name) -> bool"""
        return _engine_internal.Region_getParameterBool(self, *args, **kwargs)

    def setParameterInt32(self, *args, **kwargs):
        """setParameterInt32(self, name, value)"""
        return _engine_internal.Region_setParameterInt32(self, *args, **kwargs)

    def setParameterUInt32(self, *args, **kwargs):
        """setParameterUInt32(self, name, value)"""
        return _engine_internal.Region_setParameterUInt32(self, *args, **kwargs)

    def setParameterInt64(self, *args, **kwargs):
        """setParameterInt64(self, name, value)"""
        return _engine_internal.Region_setParameterInt64(self, *args, **kwargs)

    def setParameterUInt64(self, *args, **kwargs):
        """setParameterUInt64(self, name, value)"""
        return _engine_internal.Region_setParameterUInt64(self, *args, **kwargs)

    def setParameterReal32(self, *args, **kwargs):
        """setParameterReal32(self, name, value)"""
        return _engine_internal.Region_setParameterReal32(self, *args, **kwargs)

    def setParameterReal64(self, *args, **kwargs):
        """setParameterReal64(self, name, value)"""
        return _engine_internal.Region_setParameterReal64(self, *args, **kwargs)

    def setParameterHandle(self, *args, **kwargs):
        """setParameterHandle(self, name, value)"""
        return _engine_internal.Region_setParameterHandle(self, *args, **kwargs)

    def setParameterBool(self, *args, **kwargs):
        """setParameterBool(self, name, value)"""
        return _engine_internal.Region_setParameterBool(self, *args, **kwargs)

    def getParameterArray(self, *args, **kwargs):
        """getParameterArray(self, name, array)"""
        return _engine_internal.Region_getParameterArray(self, *args, **kwargs)

    def setParameterArray(self, *args, **kwargs):
        """setParameterArray(self, name, array)"""
        return _engine_internal.Region_setParameterArray(self, *args, **kwargs)

    def setParameterString(self, *args, **kwargs):
        """setParameterString(self, name, s)"""
        return _engine_internal.Region_setParameterString(self, *args, **kwargs)

    def getParameterString(self, *args, **kwargs):
        """getParameterString(self, name) -> std::string"""
        return _engine_internal.Region_getParameterString(self, *args, **kwargs)

    def isParameterShared(self, *args, **kwargs):
        """isParameterShared(self, name) -> bool"""
        return _engine_internal.Region_isParameterShared(self, *args, **kwargs)

    def prepareInputs(self):
        """prepareInputs(self)"""
        return _engine_internal.Region_prepareInputs(self)

    def enable(self):
        """enable(self)"""
        return _engine_internal.Region_enable(self)

    def disable(self):
        """disable(self)"""
        return _engine_internal.Region_disable(self)

    def executeCommand(self, *args, **kwargs):
        """executeCommand(self, args) -> std::string"""
        return _engine_internal.Region_executeCommand(self, *args, **kwargs)

    def compute(self):
        """compute(self)"""
        return _engine_internal.Region_compute(self)

    def enableProfiling(self):
        """enableProfiling(self)"""
        return _engine_internal.Region_enableProfiling(self)

    def disableProfiling(self):
        """disableProfiling(self)"""
        return _engine_internal.Region_disableProfiling(self)

    def resetProfiling(self):
        """resetProfiling(self)"""
        return _engine_internal.Region_resetProfiling(self)

    def getComputeTimer(self):
        """getComputeTimer(self) -> Timer"""
        return _engine_internal.Region_getComputeTimer(self)

    def getExecuteTimer(self):
        """getExecuteTimer(self) -> Timer"""
        return _engine_internal.Region_getExecuteTimer(self)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.Region___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.Region___ne__(self, *args, **kwargs)

    def __init__(self, *args): 
        """
        __init__(self, name, type, nodeParams, network=None) -> Region
        __init__(self, network=None) -> Region
        """
        this = _engine_internal.new_Region(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_Region
    def initialize(self):
        """initialize(self)"""
        return _engine_internal.Region_initialize(self)

    def isInitialized(self):
        """isInitialized(self) -> bool"""
        return _engine_internal.Region_isInitialized(self)

    def getOutput(self, *args, **kwargs):
        """getOutput(self, name) -> nupic::Output *"""
        return _engine_internal.Region_getOutput(self, *args, **kwargs)

    def getInput(self, *args, **kwargs):
        """getInput(self, name) -> nupic::Input *"""
        return _engine_internal.Region_getInput(self, *args, **kwargs)

    def getInputs(self):
        """getInputs(self) -> std::map< std::string,nupic::Input *,std::less< std::string >,std::allocator< std::pair< std::string const,nupic::Input * > > > const &"""
        return _engine_internal.Region_getInputs(self)

    def getOutputs(self):
        """getOutputs(self) -> std::map< std::string,nupic::Output *,std::less< std::string >,std::allocator< std::pair< std::string const,nupic::Output * > > > const &"""
        return _engine_internal.Region_getOutputs(self)

    def evaluateLinks(self):
        """evaluateLinks(self) -> size_t"""
        return _engine_internal.Region_evaluateLinks(self)

    def getLinkErrors(self):
        """getLinkErrors(self) -> std::string"""
        return _engine_internal.Region_getLinkErrors(self)

    def getNodeOutputElementCount(self, *args, **kwargs):
        """getNodeOutputElementCount(self, name) -> size_t"""
        return _engine_internal.Region_getNodeOutputElementCount(self, *args, **kwargs)

    def initOutputs(self):
        """initOutputs(self)"""
        return _engine_internal.Region_initOutputs(self)

    def initInputs(self):
        """initInputs(self)"""
        return _engine_internal.Region_initInputs(self)

    def intialize(self):
        """intialize(self)"""
        return _engine_internal.Region_intialize(self)

    def setDimensionInfo(self, *args, **kwargs):
        """setDimensionInfo(self, info)"""
        return _engine_internal.Region_setDimensionInfo(self, *args, **kwargs)

    def getDimensionInfo(self):
        """getDimensionInfo(self) -> std::string const &"""
        return _engine_internal.Region_getDimensionInfo(self)

    def hasOutgoingLinks(self):
        """hasOutgoingLinks(self) -> bool"""
        return _engine_internal.Region_hasOutgoingLinks(self)

    def uninitialize(self):
        """uninitialize(self)"""
        return _engine_internal.Region_uninitialize(self)

    def removeAllIncomingLinks(self):
        """removeAllIncomingLinks(self)"""
        return _engine_internal.Region_removeAllIncomingLinks(self)

    def getEnabledNodes(self):
        """getEnabledNodes(self) -> nupic::NodeSet const &"""
        return _engine_internal.Region_getEnabledNodes(self)

    def setPhases(self, *args, **kwargs):
        """setPhases(self, phases)"""
        return _engine_internal.Region_setPhases(self, *args, **kwargs)

    def getPhases(self):
        """getPhases(self) -> UInt32Set"""
        return _engine_internal.Region_getPhases(self)

    def save(self, *args, **kwargs):
        """save(self, stream)"""
        return _engine_internal.Region_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, stream)"""
        return _engine_internal.Region_load(self, *args, **kwargs)

    def getSelf(self):
        """getSelf(self) -> PyObject *"""
        return _engine_internal.Region_getSelf(self)

    def getInputArray(self, *args, **kwargs):
        """getInputArray(self, name) -> PyObject *"""
        return _engine_internal.Region_getInputArray(self, *args, **kwargs)

    def getOutputArray(self, *args, **kwargs):
        """getOutputArray(self, name) -> PyObject *"""
        return _engine_internal.Region_getOutputArray(self, *args, **kwargs)

Region_swigregister = _engine_internal.Region_swigregister
Region_swigregister(Region)

def Region_getSpecFromType(*args, **kwargs):
  """Region_getSpecFromType(nodeType) -> Spec"""
  return _engine_internal.Region_getSpecFromType(*args, **kwargs)

def Region_registerPyRegion(*args, **kwargs):
  """Region_registerPyRegion(module, className)"""
  return _engine_internal.Region_registerPyRegion(*args, **kwargs)

def Region_registerCPPRegion(*args, **kwargs):
  """Region_registerCPPRegion(name, wrapper)"""
  return _engine_internal.Region_registerCPPRegion(*args, **kwargs)

def Region_unregisterPyRegion(*args, **kwargs):
  """Region_unregisterPyRegion(className)"""
  return _engine_internal.Region_unregisterPyRegion(*args, **kwargs)

def Region_unregisterCPPRegion(*args, **kwargs):
  """Region_unregisterCPPRegion(name)"""
  return _engine_internal.Region_unregisterCPPRegion(*args, **kwargs)

parameter = _engine_internal.parameter
output = _engine_internal.output
class watchData(object):
    """Proxy of C++ nupic::watchData class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    watchID = _swig_property(_engine_internal.watchData_watchID_get, _engine_internal.watchData_watchID_set)
    varName = _swig_property(_engine_internal.watchData_varName_get, _engine_internal.watchData_varName_set)
    wType = _swig_property(_engine_internal.watchData_wType_get, _engine_internal.watchData_wType_set)
    output = _swig_property(_engine_internal.watchData_output_get, _engine_internal.watchData_output_set)
    regionName = _swig_property(_engine_internal.watchData_regionName_get, _engine_internal.watchData_regionName_set)
    region = _swig_property(_engine_internal.watchData_region_get, _engine_internal.watchData_region_set)
    nodeIndex = _swig_property(_engine_internal.watchData_nodeIndex_get, _engine_internal.watchData_nodeIndex_set)
    varType = _swig_property(_engine_internal.watchData_varType_get, _engine_internal.watchData_varType_set)
    nodeName = _swig_property(_engine_internal.watchData_nodeName_get, _engine_internal.watchData_nodeName_set)
    array = _swig_property(_engine_internal.watchData_array_get, _engine_internal.watchData_array_set)
    isArray = _swig_property(_engine_internal.watchData_isArray_get, _engine_internal.watchData_isArray_set)
    sparseOutput = _swig_property(_engine_internal.watchData_sparseOutput_get, _engine_internal.watchData_sparseOutput_set)
    def __init__(self): 
        """__init__(self) -> watchData"""
        this = _engine_internal.new_watchData()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_watchData
watchData_swigregister = _engine_internal.watchData_swigregister
watchData_swigregister(watchData)

class allData(object):
    """Proxy of C++ nupic::allData class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    outStream = _swig_property(_engine_internal.allData_outStream_get, _engine_internal.allData_outStream_set)
    fileName = _swig_property(_engine_internal.allData_fileName_get, _engine_internal.allData_fileName_set)
    watches = _swig_property(_engine_internal.allData_watches_get, _engine_internal.allData_watches_set)
    def __init__(self): 
        """__init__(self) -> allData"""
        this = _engine_internal.new_allData()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_allData
allData_swigregister = _engine_internal.allData_swigregister
allData_swigregister(allData)

class Watcher(object):
    """Proxy of C++ nupic::Watcher class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        """__init__(self, fileName) -> Watcher"""
        this = _engine_internal.new_Watcher(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_Watcher
    def watchParam(self, *args, **kwargs):
        """watchParam(self, regionName, varName, nodeIndex=-1, sparseOutput=True) -> unsigned int"""
        return _engine_internal.Watcher_watchParam(self, *args, **kwargs)

    def watchOutput(self, *args, **kwargs):
        """watchOutput(self, regionName, varName, sparseOutput=True) -> unsigned int"""
        return _engine_internal.Watcher_watchOutput(self, *args, **kwargs)

    def watcherCallback(*args, **kwargs):
        """watcherCallback(net, iteration, dataIn)"""
        return _engine_internal.Watcher_watcherCallback(*args, **kwargs)

    watcherCallback = staticmethod(watcherCallback)
    def attachToNetwork(self, *args, **kwargs):
        """attachToNetwork(self, arg2)"""
        return _engine_internal.Watcher_attachToNetwork(self, *args, **kwargs)

    def detachFromNetwork(self, *args, **kwargs):
        """detachFromNetwork(self, arg2)"""
        return _engine_internal.Watcher_detachFromNetwork(self, *args, **kwargs)

    def closeFile(self):
        """closeFile(self)"""
        return _engine_internal.Watcher_closeFile(self)

    def flushFile(self):
        """flushFile(self)"""
        return _engine_internal.Watcher_flushFile(self)

Watcher_swigregister = _engine_internal.Watcher_swigregister
Watcher_swigregister(Watcher)

def Watcher_watcherCallback(*args, **kwargs):
  """Watcher_watcherCallback(net, iteration, dataIn)"""
  return _engine_internal.Watcher_watcherCallback(*args, **kwargs)

class InputSpec(object):
    """Proxy of C++ nupic::InputSpec class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> InputSpec
        __init__(self, description, dataType, count, required, regionLevel, isDefaultInput, requireSplitterMap=True, 
            sparse=False) -> InputSpec
        """
        this = _engine_internal.new_InputSpec(*args)
        try: self.this.append(this)
        except: self.this = this
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.InputSpec___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.InputSpec___ne__(self, *args, **kwargs)

    description = _swig_property(_engine_internal.InputSpec_description_get, _engine_internal.InputSpec_description_set)
    dataType = _swig_property(_engine_internal.InputSpec_dataType_get, _engine_internal.InputSpec_dataType_set)
    count = _swig_property(_engine_internal.InputSpec_count_get, _engine_internal.InputSpec_count_set)
    required = _swig_property(_engine_internal.InputSpec_required_get, _engine_internal.InputSpec_required_set)
    regionLevel = _swig_property(_engine_internal.InputSpec_regionLevel_get, _engine_internal.InputSpec_regionLevel_set)
    isDefaultInput = _swig_property(_engine_internal.InputSpec_isDefaultInput_get, _engine_internal.InputSpec_isDefaultInput_set)
    requireSplitterMap = _swig_property(_engine_internal.InputSpec_requireSplitterMap_get, _engine_internal.InputSpec_requireSplitterMap_set)
    sparse = _swig_property(_engine_internal.InputSpec_sparse_get, _engine_internal.InputSpec_sparse_set)
    __swig_destroy__ = _engine_internal.delete_InputSpec
InputSpec_swigregister = _engine_internal.InputSpec_swigregister
InputSpec_swigregister(InputSpec)

class OutputSpec(object):
    """Proxy of C++ nupic::OutputSpec class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> OutputSpec
        __init__(self, description, dataType, count, regionLevel, isDefaultOutput, sparse=False) -> OutputSpec
        """
        this = _engine_internal.new_OutputSpec(*args)
        try: self.this.append(this)
        except: self.this = this
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.OutputSpec___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.OutputSpec___ne__(self, *args, **kwargs)

    description = _swig_property(_engine_internal.OutputSpec_description_get, _engine_internal.OutputSpec_description_set)
    dataType = _swig_property(_engine_internal.OutputSpec_dataType_get, _engine_internal.OutputSpec_dataType_set)
    count = _swig_property(_engine_internal.OutputSpec_count_get, _engine_internal.OutputSpec_count_set)
    regionLevel = _swig_property(_engine_internal.OutputSpec_regionLevel_get, _engine_internal.OutputSpec_regionLevel_set)
    isDefaultOutput = _swig_property(_engine_internal.OutputSpec_isDefaultOutput_get, _engine_internal.OutputSpec_isDefaultOutput_set)
    sparse = _swig_property(_engine_internal.OutputSpec_sparse_get, _engine_internal.OutputSpec_sparse_set)
    __swig_destroy__ = _engine_internal.delete_OutputSpec
OutputSpec_swigregister = _engine_internal.OutputSpec_swigregister
OutputSpec_swigregister(OutputSpec)

class CommandSpec(object):
    """Proxy of C++ nupic::CommandSpec class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> CommandSpec
        __init__(self, description) -> CommandSpec
        """
        this = _engine_internal.new_CommandSpec(*args)
        try: self.this.append(this)
        except: self.this = this
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.CommandSpec___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.CommandSpec___ne__(self, *args, **kwargs)

    description = _swig_property(_engine_internal.CommandSpec_description_get, _engine_internal.CommandSpec_description_set)
    __swig_destroy__ = _engine_internal.delete_CommandSpec
CommandSpec_swigregister = _engine_internal.CommandSpec_swigregister
CommandSpec_swigregister(CommandSpec)

class ParameterSpec(object):
    """Proxy of C++ nupic::ParameterSpec class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    CreateAccess = _engine_internal.ParameterSpec_CreateAccess
    ReadOnlyAccess = _engine_internal.ParameterSpec_ReadOnlyAccess
    ReadWriteAccess = _engine_internal.ParameterSpec_ReadWriteAccess
    def __init__(self, *args): 
        """
        __init__(self) -> ParameterSpec
        __init__(self, description, dataType, count, constraints, defaultValue, accessMode) -> ParameterSpec
        """
        this = _engine_internal.new_ParameterSpec(*args)
        try: self.this.append(this)
        except: self.this = this
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.ParameterSpec___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.ParameterSpec___ne__(self, *args, **kwargs)

    description = _swig_property(_engine_internal.ParameterSpec_description_get, _engine_internal.ParameterSpec_description_set)
    dataType = _swig_property(_engine_internal.ParameterSpec_dataType_get, _engine_internal.ParameterSpec_dataType_set)
    count = _swig_property(_engine_internal.ParameterSpec_count_get, _engine_internal.ParameterSpec_count_set)
    constraints = _swig_property(_engine_internal.ParameterSpec_constraints_get, _engine_internal.ParameterSpec_constraints_set)
    defaultValue = _swig_property(_engine_internal.ParameterSpec_defaultValue_get, _engine_internal.ParameterSpec_defaultValue_set)
    accessMode = _swig_property(_engine_internal.ParameterSpec_accessMode_get, _engine_internal.ParameterSpec_accessMode_set)
    __swig_destroy__ = _engine_internal.delete_ParameterSpec
ParameterSpec_swigregister = _engine_internal.ParameterSpec_swigregister
ParameterSpec_swigregister(ParameterSpec)

class Spec(object):
    """Proxy of C++ nupic::Spec class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def toString(self):
        """toString(self) -> std::string"""
        return _engine_internal.Spec_toString(self)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _engine_internal.Spec___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _engine_internal.Spec___ne__(self, *args, **kwargs)

    singleNodeOnly = _swig_property(_engine_internal.Spec_singleNodeOnly_get, _engine_internal.Spec_singleNodeOnly_set)
    description = _swig_property(_engine_internal.Spec_description_get, _engine_internal.Spec_description_set)
    inputs = _swig_property(_engine_internal.Spec_inputs_get, _engine_internal.Spec_inputs_set)
    outputs = _swig_property(_engine_internal.Spec_outputs_get, _engine_internal.Spec_outputs_set)
    commands = _swig_property(_engine_internal.Spec_commands_get, _engine_internal.Spec_commands_set)
    parameters = _swig_property(_engine_internal.Spec_parameters_get, _engine_internal.Spec_parameters_set)
    def __init__(self): 
        """__init__(self) -> Spec"""
        this = _engine_internal.new_Spec()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_Spec
Spec_swigregister = _engine_internal.Spec_swigregister
Spec_swigregister(Spec)

class Link(object):
    """Proxy of C++ nupic::Link class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def connectToNetwork(self, *args, **kwargs):
        """connectToNetwork(self, src, dest)"""
        return _engine_internal.Link_connectToNetwork(self, *args, **kwargs)

    def __init__(self, *args): 
        """
        __init__(self, linkType, linkParams, srcRegionName, destRegionName, srcOutputName="", destInputName="", 
            propagationDelay=0) -> Link
        __init__(self) -> Link
        __init__(self, linkType, linkParams, srcOutput, destInput, propagationDelay=0) -> Link
        """
        this = _engine_internal.new_Link(*args)
        try: self.this.append(this)
        except: self.this = this
    def setSrcDimensions(self, *args, **kwargs):
        """setSrcDimensions(self, dims)"""
        return _engine_internal.Link_setSrcDimensions(self, *args, **kwargs)

    def setDestDimensions(self, *args, **kwargs):
        """setDestDimensions(self, dims)"""
        return _engine_internal.Link_setDestDimensions(self, *args, **kwargs)

    def initialize(self, *args, **kwargs):
        """initialize(self, destinationOffset)"""
        return _engine_internal.Link_initialize(self, *args, **kwargs)

    __swig_destroy__ = _engine_internal.delete_Link
    def getSrcDimensions(self):
        """getSrcDimensions(self) -> Dimensions"""
        return _engine_internal.Link_getSrcDimensions(self)

    def getDestDimensions(self):
        """getDestDimensions(self) -> Dimensions"""
        return _engine_internal.Link_getDestDimensions(self)

    def getLinkType(self):
        """getLinkType(self) -> std::string const &"""
        return _engine_internal.Link_getLinkType(self)

    def getLinkParams(self):
        """getLinkParams(self) -> std::string const &"""
        return _engine_internal.Link_getLinkParams(self)

    def getSrcRegionName(self):
        """getSrcRegionName(self) -> std::string const &"""
        return _engine_internal.Link_getSrcRegionName(self)

    def getSrcOutputName(self):
        """getSrcOutputName(self) -> std::string const &"""
        return _engine_internal.Link_getSrcOutputName(self)

    def getDestRegionName(self):
        """getDestRegionName(self) -> std::string const &"""
        return _engine_internal.Link_getDestRegionName(self)

    def getDestInputName(self):
        """getDestInputName(self) -> std::string const &"""
        return _engine_internal.Link_getDestInputName(self)

    def getPropagationDelay(self):
        """getPropagationDelay(self) -> size_t"""
        return _engine_internal.Link_getPropagationDelay(self)

    def getMoniker(self):
        """getMoniker(self) -> std::string"""
        return _engine_internal.Link_getMoniker(self)

    def getSrc(self):
        """getSrc(self) -> nupic::Output &"""
        return _engine_internal.Link_getSrc(self)

    def getDest(self):
        """getDest(self) -> nupic::Input &"""
        return _engine_internal.Link_getDest(self)

    def compute(self):
        """compute(self)"""
        return _engine_internal.Link_compute(self)

    def buildSplitterMap(self, *args, **kwargs):
        """buildSplitterMap(self, splitter)"""
        return _engine_internal.Link_buildSplitterMap(self, *args, **kwargs)

    def shiftBufferedData(self):
        """shiftBufferedData(self)"""
        return _engine_internal.Link_shiftBufferedData(self)

    def toString(self):
        """toString(self) -> std::string const"""
        return _engine_internal.Link_toString(self)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, o) -> bool"""
        return _engine_internal.Link___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, o) -> bool"""
        return _engine_internal.Link___ne__(self, *args, **kwargs)

    def serialize(self, *args, **kwargs):
        """serialize(self, f)"""
        return _engine_internal.Link_serialize(self, *args, **kwargs)

    def deserialize(self, *args, **kwargs):
        """deserialize(self, f)"""
        return _engine_internal.Link_deserialize(self, *args, **kwargs)

Link_swigregister = _engine_internal.Link_swigregister
Link_swigregister(Link)

class InputPair(object):
    """Proxy of C++ std::pair<(std::string,nupic::InputSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> InputPair
        __init__(self, first, second) -> InputPair
        __init__(self, p) -> InputPair
        """
        this = _engine_internal.new_InputPair(*args)
        try: self.this.append(this)
        except: self.this = this
    first = _swig_property(_engine_internal.InputPair_first_get, _engine_internal.InputPair_first_set)
    second = _swig_property(_engine_internal.InputPair_second_get, _engine_internal.InputPair_second_set)
    def __len__(self): return 2
    def __repr__(self): return str((self.first, self.second))
    def __getitem__(self, index): 
      if not (index % 2): 
        return self.first
      else:
        return self.second
    def __setitem__(self, index, val):
      if not (index % 2): 
        self.first = val
      else:
        self.second = val
    __swig_destroy__ = _engine_internal.delete_InputPair
InputPair_swigregister = _engine_internal.InputPair_swigregister
InputPair_swigregister(InputPair)

class OutputPair(object):
    """Proxy of C++ std::pair<(std::string,nupic::OutputSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> OutputPair
        __init__(self, first, second) -> OutputPair
        __init__(self, p) -> OutputPair
        """
        this = _engine_internal.new_OutputPair(*args)
        try: self.this.append(this)
        except: self.this = this
    first = _swig_property(_engine_internal.OutputPair_first_get, _engine_internal.OutputPair_first_set)
    second = _swig_property(_engine_internal.OutputPair_second_get, _engine_internal.OutputPair_second_set)
    def __len__(self): return 2
    def __repr__(self): return str((self.first, self.second))
    def __getitem__(self, index): 
      if not (index % 2): 
        return self.first
      else:
        return self.second
    def __setitem__(self, index, val):
      if not (index % 2): 
        self.first = val
      else:
        self.second = val
    __swig_destroy__ = _engine_internal.delete_OutputPair
OutputPair_swigregister = _engine_internal.OutputPair_swigregister
OutputPair_swigregister(OutputPair)

class ParameterPair(object):
    """Proxy of C++ std::pair<(std::string,nupic::ParameterSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> ParameterPair
        __init__(self, first, second) -> ParameterPair
        __init__(self, p) -> ParameterPair
        """
        this = _engine_internal.new_ParameterPair(*args)
        try: self.this.append(this)
        except: self.this = this
    first = _swig_property(_engine_internal.ParameterPair_first_get, _engine_internal.ParameterPair_first_set)
    second = _swig_property(_engine_internal.ParameterPair_second_get, _engine_internal.ParameterPair_second_set)
    def __len__(self): return 2
    def __repr__(self): return str((self.first, self.second))
    def __getitem__(self, index): 
      if not (index % 2): 
        return self.first
      else:
        return self.second
    def __setitem__(self, index, val):
      if not (index % 2): 
        self.first = val
      else:
        self.second = val
    __swig_destroy__ = _engine_internal.delete_ParameterPair
ParameterPair_swigregister = _engine_internal.ParameterPair_swigregister
ParameterPair_swigregister(ParameterPair)

class CommandPair(object):
    """Proxy of C++ std::pair<(std::string,nupic::CommandSpec)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> CommandPair
        __init__(self, first, second) -> CommandPair
        __init__(self, p) -> CommandPair
        """
        this = _engine_internal.new_CommandPair(*args)
        try: self.this.append(this)
        except: self.this = this
    first = _swig_property(_engine_internal.CommandPair_first_get, _engine_internal.CommandPair_first_set)
    second = _swig_property(_engine_internal.CommandPair_second_get, _engine_internal.CommandPair_second_set)
    def __len__(self): return 2
    def __repr__(self): return str((self.first, self.second))
    def __getitem__(self, index): 
      if not (index % 2): 
        return self.first
      else:
        return self.second
    def __setitem__(self, index, val):
      if not (index % 2): 
        self.first = val
      else:
        self.second = val
    __swig_destroy__ = _engine_internal.delete_CommandPair
CommandPair_swigregister = _engine_internal.CommandPair_swigregister
CommandPair_swigregister(CommandPair)

class RegionPair(object):
    """Proxy of C++ std::pair<(std::string,p.nupic::Region)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> RegionPair
        __init__(self, __a, __b) -> RegionPair
        __init__(self, __p) -> RegionPair
        """
        this = _engine_internal.new_RegionPair(*args)
        try: self.this.append(this)
        except: self.this = this
    first = _swig_property(_engine_internal.RegionPair_first_get, _engine_internal.RegionPair_first_set)
    second = _swig_property(_engine_internal.RegionPair_second_get, _engine_internal.RegionPair_second_set)
    def __len__(self): return 2
    def __repr__(self): return str((self.first, self.second))
    def __getitem__(self, index): 
      if not (index % 2): 
        return self.first
      else:
        return self.second
    def __setitem__(self, index, val):
      if not (index % 2): 
        self.first = val
      else:
        self.second = val
    __swig_destroy__ = _engine_internal.delete_RegionPair
RegionPair_swigregister = _engine_internal.RegionPair_swigregister
RegionPair_swigregister(RegionPair)

class LinkPair(object):
    """Proxy of C++ std::pair<(std::string,p.nupic::Link)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> LinkPair
        __init__(self, __a, __b) -> LinkPair
        __init__(self, __p) -> LinkPair
        """
        this = _engine_internal.new_LinkPair(*args)
        try: self.this.append(this)
        except: self.this = this
    first = _swig_property(_engine_internal.LinkPair_first_get, _engine_internal.LinkPair_first_set)
    second = _swig_property(_engine_internal.LinkPair_second_get, _engine_internal.LinkPair_second_set)
    def __len__(self): return 2
    def __repr__(self): return str((self.first, self.second))
    def __getitem__(self, index): 
      if not (index % 2): 
        return self.first
      else:
        return self.second
    def __setitem__(self, index, val):
      if not (index % 2): 
        self.first = val
      else:
        self.second = val
    def __iter__(self):
      return IterablePair(self)

    __swig_destroy__ = _engine_internal.delete_LinkPair
LinkPair_swigregister = _engine_internal.LinkPair_swigregister
LinkPair_swigregister(LinkPair)

class Timer(object):
    """Proxy of C++ nupic::Timer class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, startme=False): 
        """__init__(self, startme=False) -> Timer"""
        this = _engine_internal.new_Timer(startme)
        try: self.this.append(this)
        except: self.this = this
    def start(self):
        """start(self)"""
        return _engine_internal.Timer_start(self)

    def stop(self):
        """stop(self)"""
        return _engine_internal.Timer_stop(self)

    def getElapsed(self):
        """getElapsed(self) -> nupic::Real64"""
        return _engine_internal.Timer_getElapsed(self)

    def reset(self):
        """reset(self)"""
        return _engine_internal.Timer_reset(self)

    def getStartCount(self):
        """getStartCount(self) -> nupic::UInt64"""
        return _engine_internal.Timer_getStartCount(self)

    def isStarted(self):
        """isStarted(self) -> bool"""
        return _engine_internal.Timer_isStarted(self)

    def toString(self):
        """toString(self) -> std::string"""
        return _engine_internal.Timer_toString(self)

    def __str__(self):
      return self.toString()

    elapsed = property(getElapsed)
    startCount = property(getStartCount)

    __swig_destroy__ = _engine_internal.delete_Timer
Timer_swigregister = _engine_internal.Timer_swigregister
Timer_swigregister(Timer)


def getBasicType(*args):
  """
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    getBasicType(arg1) -> NTA_BasicType
    """
  return _engine_internal.getBasicType(*args)

def array2numpy(*args, **kwargs):
  """array2numpy(a) -> PyObject *"""
  return _engine_internal.array2numpy(*args, **kwargs)
class ByteArray(Array):
    """Proxy of C++ nupic::PyArray<(nupic::Byte)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> ByteArray
        __init__(self, count) -> ByteArray
        """
        this = _engine_internal.new_ByteArray(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.ByteArray_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> char"""
        return _engine_internal.ByteArray___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.ByteArray___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.ByteArray___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.ByteArray___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.ByteArray___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.ByteArray_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_ByteArray
ByteArray_swigregister = _engine_internal.ByteArray_swigregister
ByteArray_swigregister(ByteArray)

class Int16Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::Int16)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Int16Array
        __init__(self, count) -> Int16Array
        """
        this = _engine_internal.new_Int16Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Int16Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> short"""
        return _engine_internal.Int16Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Int16Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Int16Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Int16Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Int16Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Int16Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Int16Array
Int16Array_swigregister = _engine_internal.Int16Array_swigregister
Int16Array_swigregister(Int16Array)

class UInt16Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::UInt16)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> UInt16Array
        __init__(self, count) -> UInt16Array
        """
        this = _engine_internal.new_UInt16Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.UInt16Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> unsigned short"""
        return _engine_internal.UInt16Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.UInt16Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.UInt16Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.UInt16Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.UInt16Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.UInt16Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_UInt16Array
UInt16Array_swigregister = _engine_internal.UInt16Array_swigregister
UInt16Array_swigregister(UInt16Array)

class Int32Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::Int32)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Int32Array
        __init__(self, count) -> Int32Array
        """
        this = _engine_internal.new_Int32Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Int32Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> int"""
        return _engine_internal.Int32Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Int32Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Int32Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Int32Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Int32Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Int32Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Int32Array
Int32Array_swigregister = _engine_internal.Int32Array_swigregister
Int32Array_swigregister(Int32Array)

class UInt32Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::UInt32)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> UInt32Array
        __init__(self, count) -> UInt32Array
        """
        this = _engine_internal.new_UInt32Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.UInt32Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> unsigned int"""
        return _engine_internal.UInt32Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.UInt32Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.UInt32Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.UInt32Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.UInt32Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.UInt32Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_UInt32Array
UInt32Array_swigregister = _engine_internal.UInt32Array_swigregister
UInt32Array_swigregister(UInt32Array)

class Int64Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::Int64)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Int64Array
        __init__(self, count) -> Int64Array
        """
        this = _engine_internal.new_Int64Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Int64Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> long"""
        return _engine_internal.Int64Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Int64Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Int64Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Int64Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Int64Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Int64Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Int64Array
Int64Array_swigregister = _engine_internal.Int64Array_swigregister
Int64Array_swigregister(Int64Array)

class UInt64Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::UInt64)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> UInt64Array
        __init__(self, count) -> UInt64Array
        """
        this = _engine_internal.new_UInt64Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.UInt64Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> unsigned long"""
        return _engine_internal.UInt64Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.UInt64Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.UInt64Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.UInt64Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.UInt64Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.UInt64Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_UInt64Array
UInt64Array_swigregister = _engine_internal.UInt64Array_swigregister
UInt64Array_swigregister(UInt64Array)

class Real32Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::Real32)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Real32Array
        __init__(self, count) -> Real32Array
        """
        this = _engine_internal.new_Real32Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Real32Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> float"""
        return _engine_internal.Real32Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Real32Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Real32Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Real32Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Real32Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Real32Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Real32Array
Real32Array_swigregister = _engine_internal.Real32Array_swigregister
Real32Array_swigregister(Real32Array)

class Real64Array(Array):
    """Proxy of C++ nupic::PyArray<(nupic::Real64)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Real64Array
        __init__(self, count) -> Real64Array
        """
        this = _engine_internal.new_Real64Array(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Real64Array_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> double"""
        return _engine_internal.Real64Array___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Real64Array___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Real64Array___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Real64Array___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Real64Array___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Real64Array_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Real64Array
Real64Array_swigregister = _engine_internal.Real64Array_swigregister
Real64Array_swigregister(Real64Array)

class BoolArray(Array):
    """Proxy of C++ nupic::PyArray<(bool)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> BoolArray
        __init__(self, count) -> BoolArray
        """
        this = _engine_internal.new_BoolArray(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.BoolArray_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> bool"""
        return _engine_internal.BoolArray___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.BoolArray___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.BoolArray___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.BoolArray___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.BoolArray___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.BoolArray_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_BoolArray
BoolArray_swigregister = _engine_internal.BoolArray_swigregister
BoolArray_swigregister(BoolArray)

class ByteArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::Byte)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> ByteArrayRef
        __init__(self, a) -> ByteArrayRef
        """
        this = _engine_internal.new_ByteArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.ByteArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> char"""
        return _engine_internal.ByteArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.ByteArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.ByteArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.ByteArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.ByteArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.ByteArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_ByteArrayRef
ByteArrayRef_swigregister = _engine_internal.ByteArrayRef_swigregister
ByteArrayRef_swigregister(ByteArrayRef)

class Int16ArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::Int16)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Int16ArrayRef
        __init__(self, a) -> Int16ArrayRef
        """
        this = _engine_internal.new_Int16ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Int16ArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> short"""
        return _engine_internal.Int16ArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Int16ArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Int16ArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Int16ArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Int16ArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Int16ArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Int16ArrayRef
Int16ArrayRef_swigregister = _engine_internal.Int16ArrayRef_swigregister
Int16ArrayRef_swigregister(Int16ArrayRef)

class UInt16ArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::UInt16)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> UInt16ArrayRef
        __init__(self, a) -> UInt16ArrayRef
        """
        this = _engine_internal.new_UInt16ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.UInt16ArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> unsigned short"""
        return _engine_internal.UInt16ArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.UInt16ArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.UInt16ArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.UInt16ArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.UInt16ArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.UInt16ArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_UInt16ArrayRef
UInt16ArrayRef_swigregister = _engine_internal.UInt16ArrayRef_swigregister
UInt16ArrayRef_swigregister(UInt16ArrayRef)

class Int32ArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::Int32)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Int32ArrayRef
        __init__(self, a) -> Int32ArrayRef
        """
        this = _engine_internal.new_Int32ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Int32ArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> int"""
        return _engine_internal.Int32ArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Int32ArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Int32ArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Int32ArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Int32ArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Int32ArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Int32ArrayRef
Int32ArrayRef_swigregister = _engine_internal.Int32ArrayRef_swigregister
Int32ArrayRef_swigregister(Int32ArrayRef)

class UInt32ArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::UInt32)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> UInt32ArrayRef
        __init__(self, a) -> UInt32ArrayRef
        """
        this = _engine_internal.new_UInt32ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.UInt32ArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> unsigned int"""
        return _engine_internal.UInt32ArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.UInt32ArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.UInt32ArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.UInt32ArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.UInt32ArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.UInt32ArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_UInt32ArrayRef
UInt32ArrayRef_swigregister = _engine_internal.UInt32ArrayRef_swigregister
UInt32ArrayRef_swigregister(UInt32ArrayRef)

class Int64ArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::Int64)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Int64ArrayRef
        __init__(self, a) -> Int64ArrayRef
        """
        this = _engine_internal.new_Int64ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Int64ArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> long"""
        return _engine_internal.Int64ArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Int64ArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Int64ArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Int64ArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Int64ArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Int64ArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Int64ArrayRef
Int64ArrayRef_swigregister = _engine_internal.Int64ArrayRef_swigregister
Int64ArrayRef_swigregister(Int64ArrayRef)

class UInt64ArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::UInt64)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> UInt64ArrayRef
        __init__(self, a) -> UInt64ArrayRef
        """
        this = _engine_internal.new_UInt64ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.UInt64ArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> unsigned long"""
        return _engine_internal.UInt64ArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.UInt64ArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.UInt64ArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.UInt64ArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.UInt64ArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.UInt64ArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_UInt64ArrayRef
UInt64ArrayRef_swigregister = _engine_internal.UInt64ArrayRef_swigregister
UInt64ArrayRef_swigregister(UInt64ArrayRef)

class Real32ArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(nupic::Real32)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> Real32ArrayRef
        __init__(self, a) -> Real32ArrayRef
        """
        this = _engine_internal.new_Real32ArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.Real32ArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> float"""
        return _engine_internal.Real32ArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.Real32ArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.Real32ArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.Real32ArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.Real32ArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.Real32ArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_Real32ArrayRef
Real32ArrayRef_swigregister = _engine_internal.Real32ArrayRef_swigregister
Real32ArrayRef_swigregister(Real32ArrayRef)

class BoolArrayRef(ArrayRef):
    """Proxy of C++ nupic::PyArrayRef<(bool)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args): 
        """
        __init__(self) -> BoolArrayRef
        __init__(self, a) -> BoolArrayRef
        """
        this = _engine_internal.new_BoolArrayRef(*args)
        try: self.this.append(this)
        except: self.this = this
    def getType(self):
        """getType(self) -> NTA_BasicType"""
        return _engine_internal.BoolArrayRef_getType(self)

    def __getitem__(self, *args, **kwargs):
        """__getitem__(self, i) -> bool"""
        return _engine_internal.BoolArrayRef___getitem__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """__setitem__(self, i, x)"""
        return _engine_internal.BoolArrayRef___setitem__(self, *args, **kwargs)

    def __len__(self):
        """__len__(self) -> size_t"""
        return _engine_internal.BoolArrayRef___len__(self)

    def __repr__(self):
        """__repr__(self) -> std::string"""
        return _engine_internal.BoolArrayRef___repr__(self)

    def __str__(self):
        """__str__(self) -> std::string"""
        return _engine_internal.BoolArrayRef___str__(self)

    def asNumpyArray(self):
        """asNumpyArray(self) -> PyObject *"""
        return _engine_internal.BoolArrayRef_asNumpyArray(self)

    __swig_destroy__ = _engine_internal.delete_BoolArrayRef
BoolArrayRef_swigregister = _engine_internal.BoolArrayRef_swigregister
BoolArrayRef_swigregister(BoolArrayRef)

class OS(object):
    """Proxy of C++ nupic::OS class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def getProcessMemoryUsage():
        """getProcessMemoryUsage()"""
        return _engine_internal.OS_getProcessMemoryUsage()

    getProcessMemoryUsage = staticmethod(getProcessMemoryUsage)
    def __init__(self): 
        """__init__(self) -> OS"""
        this = _engine_internal.new_OS()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _engine_internal.delete_OS
OS_swigregister = _engine_internal.OS_swigregister
OS_swigregister(OS)

def OS_getProcessMemoryUsage():
  """OS_getProcessMemoryUsage()"""
  return _engine_internal.OS_getProcessMemoryUsage()



