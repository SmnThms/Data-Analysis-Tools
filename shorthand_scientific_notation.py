# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:14:30 2023, @author: Simon
"""
import numpy as np

def short(value,uncertainty=None,
          u_digits=2,exponent_separator='x10^',decimal_separator='.'):
        """
        Returns a string in shorthand scientific notation:
            mantissa (last digit uncertainty) x10^ exponent
        
        Optional arguments:
        > uncertainty (default=None)
            no parenthesis is included if no uncertainty is provided.
        > u_digits (default=2)
            the number of digits in the parenthesis.
        > exponent_separator (default='x10^')
            notation introducing the exponent ('x10^','E','e'…).
        > decimal_separator (default='.').
        """
        exponent = lambda x:int(np.floor(np.log10(abs(x)))) if x!=0 else 0
        separator = lambda s,i:decimal_separator.join([s[:i],s[i:]])        
        if uncertainty is None: uncertainty = 0
        if value!=0:
            exp_v, exp_u = exponent(value), exponent(uncertainty)
            mantissa = lambda x:str(round(x*10**(-min(exp_v,exp_u)+u_digits-1)))
            val_str, unc_str = mantissa(value), mantissa(abs(uncertainty))
            if exp_u<exp_v or u_digits!=1:
                val_str = separator(val_str,{True:1,False:2}[value>0])
            if u_digits!=1:
                if exp_u==exp_v: unc_str = separator(unc_str,1)
                elif exp_u>exp_v: unc_str = separator(unc_str,1-u_digits)
        else:
            val_str, exp_v = '0', exponent(uncertainty)
            unc_str = str(round(abs(uncertainty)*10**(-exp_v+u_digits-1)))
            if u_digits!=1: unc_str = separator(unc_str,1)
        if uncertainty!=0: val_str += '(' + unc_str + ')'
        if exp_v!=0: val_str += exponent_separator + str(exp_v)
        return val_str
    
def measure(dictionary,key,eqsep=' = ',u_digits=2,expsep='E',decsep='.'):
    """
	Returns the shorthand scientific notation of dictionary[key],
    taking dictionary['u_'+key] as the uncertainty if such an entry exists,
    followed by dictionary['units'][key] if such an entry exists,
    and preceded by the variable name, key+eqsep (if eqsep!=None).

    Other optional arguments: as in short.
    """
    s = short(dictionary[key],dictionary.get('u_'+key),u_digits,expsep,decsep)
    if 'units' in dictionary and isinstance(dictionary['units'],dict): 
        s += ' '+dictionary['units'].get(key,'')
    return key+eqsep+s if eqsep is not None else s

# EXAMPLE #####################################################################
# d = {'truc':-13.6e-3,'u_truc':0.34e-3,'units':{'truc':'keV'}}
# print(measure(d,'truc'))