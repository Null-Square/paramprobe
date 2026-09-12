import os
from pathlib import Path
import numpy as np
import pytest
import torch
from paramprobe.compact_kv import (encode_block,decode_block,export_store,FileKV,top_mass_indices,HEADER)
from paramprobe.payload_theory import operator,pruning_bound,tanh_via_paired_softmax

@pytest.mark.parametrize('codec,bytes_', [('fp32',16384),('fp16',8192),('int8',4096)])
def test_full_21_slot_roundtrip(codec,bytes_):
    g=np.random.default_rng(4); k=g.normal(size=(21,96)).astype('float32'); v=g.normal(size=k.shape).astype('float32')
    block=encode_block(k,v,bytes_,codec); assert len(block)==bytes_ and HEADER.size==32
    a,b=decode_block(block)
    atol={'fp32':0,'fp16':.002,'int8':max(abs(k).max(),abs(v).max())/254+1e-6}[codec]
    np.testing.assert_allclose(a,k,atol=atol,rtol=0); np.testing.assert_allclose(b,v,atol=atol,rtol=0)

@pytest.mark.parametrize('n,bytes_',[(5,4096),(10,8192),(21,16384)])
def test_float_page_sizes(n,bytes_):
    assert len(encode_block(np.zeros((n,96)),np.zeros((n,96)),bytes_))==bytes_

@pytest.mark.parametrize('codec',['fp32','fp16','int8'])
def test_overflow_refused(codec):
    with pytest.raises(ValueError): encode_block(np.ones((21,96)),np.ones((21,96)),100,codec)

@pytest.mark.parametrize('bad',[0,-1,22,1.2,True])
def test_bad_selection(bad):
    with pytest.raises((TypeError,ValueError)): top_mass_indices(np.ones((2,21)),bad)

def test_tie_break_and_mass_validation():
    np.testing.assert_array_equal(top_mass_indices(np.ones((2,21)),3),[[0,1,2],[0,1,2]])
    for x in [np.array([[np.nan]]),np.array([[-1.]])]:
        with pytest.raises(ValueError): top_mass_indices(x,1)

@pytest.mark.parametrize('size',[4.9,True])
def test_fractional_blocks_refused(size):
    with pytest.raises(TypeError): encode_block(np.ones((2,2)),np.ones((2,2)),size)

def test_bad_blocks():
    valid=encode_block(np.ones((2,2)),np.ones((2,2)),128)
    for b in [b'',b'bad'+valid[3:],valid[:31],valid[:40]]:
        with pytest.raises(ValueError): decode_block(b)
    with pytest.raises(ValueError): encode_block(np.array([[np.nan]]),np.ones((1,1)),128)
    with pytest.raises(ValueError): encode_block(np.full((2,2),1e10),np.ones((2,2)),128,'fp16')

def test_file_reads_duplicates_close_and_empty(tmp_path):
    k=np.ones((2,3,4),np.float32); v=k.copy(); p=tmp_path/'p'
    export_store(p,k,v,np.tile(np.arange(3),(2,1)),128,'fp32')
    with FileKV(p,128,4) as reader:
        out=reader.residuals(np.array([1,1,0]),np.ones((3,4)))
        assert reader.reads==3 and reader.returned_bytes==384
        np.testing.assert_allclose(out,.15*np.tanh(1),atol=1e-7)
        with pytest.raises(IndexError): reader.residuals(np.array([2]),np.ones((1,4)))
        assert reader.reads==3
    with pytest.raises(RuntimeError): reader.residuals(np.array([0]),np.ones((1,4)))
    e=tmp_path/'empty'; e.touch()
    with pytest.raises(ValueError): FileKV(e,128,4)
    with pytest.raises(FileExistsError): export_store(p,k,v,np.tile(np.arange(3),(2,1)),128,'fp32')

def test_short_read_is_accounted(tmp_path):
    p=tmp_path/'p'; p.write_bytes(encode_block(np.ones((2,4)),np.ones((2,4)),128))
    with FileKV(p,128,4) as r:
        p.write_bytes(b'12')
        with pytest.raises(OSError): r.residuals(np.array([0]),np.ones((1,4)))
        assert (r.reads,r.returned_bytes,r.failed_reads)==(1,2,1)

@pytest.mark.parametrize('seed',range(10))
def test_tanh_operator_equals_tied_two_slot_memory_forward_and_gradient(seed):
    torch.manual_seed(seed)
    tensors=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in [(8,96),(20,96),(20,),(96,20),(96,)]]
    a=operator(*tensors); b=operator(*tensors,paired=True)
    torch.testing.assert_close(a,b,atol=2e-14,rtol=2e-13)
    weights=torch.randn_like(a)
    ga=torch.autograd.grad((a*weights).sum(),tensors,retain_graph=True)
    gb=torch.autograd.grad((b*weights).sum(),tensors)
    for x,y in zip(ga,gb): torch.testing.assert_close(x,y,atol=2e-13,rtol=1e-11)

def test_extreme_logits():
    z=torch.tensor([-1000.,-100.,0.,100.,1000.],dtype=torch.float64)
    torch.testing.assert_close(torch.tanh(z),tanh_via_paired_softmax(z),atol=1e-14,rtol=0)

@pytest.mark.parametrize('seed',range(10))
def test_pruning_bound(seed):
    torch.manual_seed(seed)
    p=torch.randn(80,21,dtype=torch.float64).softmax(-1)
    v=torch.randn(21,96,dtype=torch.float64)
    for n in [1,5,10,21]:
        kept=torch.arange(n); q=p[:,kept]/p[:,kept].sum(-1,keepdim=True)
        error=torch.linalg.vector_norm(.15*(torch.tanh(p@v)-torch.tanh(q@v[kept])),dim=-1)
        assert bool((error<=pruning_bound(p,v,kept)+1e-13).all())

def test_count_of_hard_winners_is_not_functional_rank():
    # Every query has the same hard winner, yet the probability rows are independent.
    p=torch.tensor([[.6,.3,.1],[.6,.1,.3],[.8,.1,.1]],dtype=torch.float64)
    assert p.argmax(-1).unique().numel()==1
    assert torch.linalg.matrix_rank(p)==3
