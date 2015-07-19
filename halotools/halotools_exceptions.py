# -*- coding: utf-8 -*-
"""
Classes for all package-specific exceptions. 
"""
__all__ = ['HalotoolsError', 'HalotoolsCacheError']

class HalotoolsError(Exception):
	pass

class HalotoolsCacheError(HalotoolsError):
	pass


