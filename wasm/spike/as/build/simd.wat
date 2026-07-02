(module
 (type $0 (func (param i32) (result i32)))
 (type $1 (func (param i32 i32 i32 i32)))
 (type $2 (func (param i32 i32 i32) (result f64)))
 (type $3 (func (param i32 f64 i32 i32)))
 (global $assembly/simd/_bump (mut i32) (i32.const 0))
 (memory $0 0)
 (export "alloc" (func $assembly/simd/alloc))
 (export "add_f64" (func $assembly/simd/add_f64))
 (export "sum_f64_null" (func $assembly/simd/sum_f64_null))
 (export "cmp_gt_f64_mask" (func $assembly/simd/cmp_gt_f64_mask))
 (export "memory" (memory $0))
 (func $assembly/simd/alloc (param $0 i32) (result i32)
  (local $1 i32)
  global.get $assembly/simd/_bump
  i32.eqz
  if
   i32.const 1024
   global.set $assembly/simd/_bump
  end
  global.get $assembly/simd/_bump
  local.set $1
  global.get $assembly/simd/_bump
  local.get $0
  i32.const 7
  i32.add
  i32.const -8
  i32.and
  i32.add
  global.set $assembly/simd/_bump
  global.get $assembly/simd/_bump
  i32.const 65535
  i32.add
  i32.const 16
  i32.shr_u
  local.tee $0
  memory.size
  i32.gt_s
  if
   local.get $0
   memory.size
   i32.sub
   memory.grow
   drop
  end
  local.get $1
 )
 (func $assembly/simd/add_f64 (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32)
  (local $4 i32)
  (local $5 i32)
  loop $while-continue|0
   local.get $4
   i32.const 1
   i32.add
   local.get $3
   i32.lt_u
   if
    local.get $4
    i32.const 3
    i32.shl
    local.tee $5
    local.get $2
    i32.add
    local.get $0
    local.get $5
    i32.add
    v128.load
    local.get $1
    local.get $5
    i32.add
    v128.load
    f64x2.add
    v128.store
    local.get $4
    i32.const 2
    i32.add
    local.set $4
    br $while-continue|0
   end
  end
  local.get $3
  local.get $4
  i32.gt_u
  if
   local.get $4
   i32.const 3
   i32.shl
   local.tee $3
   local.get $2
   i32.add
   local.get $0
   local.get $3
   i32.add
   f64.load
   local.get $1
   local.get $3
   i32.add
   f64.load
   f64.add
   f64.store
  end
 )
 (func $assembly/simd/sum_f64_null (param $0 i32) (param $1 i32) (param $2 i32) (result f64)
  (local $3 f64)
  (local $4 i32)
  (local $5 v128)
  (local $6 v128)
  (local $7 i64)
  (local $8 i64)
  (local $9 i64)
  (local $10 i64)
  (local $11 i32)
  (local $12 i32)
  loop $while-continue|0
   local.get $4
   i32.const 3
   i32.add
   local.get $2
   i32.lt_u
   if
    local.get $4
    i32.const 3
    i32.shr_u
    local.set $11
    local.get $4
    i32.const 7
    i32.and
    local.tee $12
    i32.const 5
    i32.lt_u
    if (result i64)
     i64.const -1
     i64.const 0
     local.get $1
     local.get $11
     i32.add
     i32.load8_u
     local.tee $11
     local.get $12
     i32.shr_u
     i32.const 1
     i32.and
     select
     local.set $7
     i64.const -1
     i64.const 0
     local.get $11
     local.get $12
     i32.const 1
     i32.add
     i32.shr_u
     i32.const 1
     i32.and
     select
     local.set $8
     i64.const -1
     i64.const 0
     local.get $11
     local.get $12
     i32.const 2
     i32.add
     i32.shr_u
     i32.const 1
     i32.and
     select
     local.set $9
     i64.const -1
     i64.const 0
     local.get $11
     local.get $12
     i32.const 3
     i32.add
     i32.shr_u
     i32.const 1
     i32.and
     select
    else
     local.get $12
     i32.const 5
     i32.eq
     if (result i64)
      i64.const -1
      i64.const 0
      local.get $1
      local.get $11
      i32.add
      local.tee $11
      i32.load8_u
      local.tee $12
      i32.const 5
      i32.shr_u
      i32.const 1
      i32.and
      select
      local.set $7
      i64.const -1
      i64.const 0
      local.get $12
      i32.const 6
      i32.shr_u
      i32.const 1
      i32.and
      select
      local.set $8
      i64.const -1
      i64.const 0
      local.get $12
      i32.const 7
      i32.shr_u
      select
      local.set $9
      i64.const -1
      i64.const 0
      local.get $11
      i32.load8_u offset=1
      i32.const 1
      i32.and
      select
     else
      local.get $12
      i32.const 6
      i32.eq
      if (result i64)
       i64.const -1
       i64.const 0
       local.get $1
       local.get $11
       i32.add
       local.tee $11
       i32.load8_u
       local.tee $12
       i32.const 6
       i32.shr_u
       i32.const 1
       i32.and
       select
       local.set $7
       i64.const -1
       i64.const 0
       local.get $12
       i32.const 7
       i32.shr_u
       select
       local.set $8
       i64.const -1
       i64.const 0
       local.get $11
       i32.load8_u offset=1
       local.tee $11
       i32.const 1
       i32.and
       select
       local.set $9
       i64.const -1
       i64.const 0
       local.get $11
       i32.const 1
       i32.shr_u
       i32.const 1
       i32.and
       select
      else
       i64.const -1
       i64.const 0
       local.get $1
       local.get $11
       i32.add
       local.tee $11
       i32.load8_u
       i32.const 7
       i32.shr_u
       select
       local.set $7
       i64.const -1
       i64.const 0
       local.get $11
       i32.load8_u offset=1
       local.tee $11
       i32.const 1
       i32.and
       select
       local.set $8
       i64.const -1
       i64.const 0
       local.get $11
       i32.const 1
       i32.shr_u
       i32.const 1
       i32.and
       select
       local.set $9
       i64.const -1
       i64.const 0
       local.get $11
       i32.const 2
       i32.shr_u
       i32.const 1
       i32.and
       select
      end
     end
    end
    local.set $10
    local.get $5
    local.get $0
    local.get $4
    i32.const 3
    i32.shl
    i32.add
    v128.load
    local.get $7
    i64x2.splat
    local.get $8
    i64x2.replace_lane 1
    v128.and
    f64x2.add
    local.set $5
    local.get $6
    local.get $0
    local.get $4
    i32.const 2
    i32.add
    i32.const 3
    i32.shl
    i32.add
    v128.load
    local.get $9
    i64x2.splat
    local.get $10
    i64x2.replace_lane 1
    v128.and
    f64x2.add
    local.set $6
    local.get $4
    i32.const 4
    i32.add
    local.set $4
    br $while-continue|0
   end
  end
  local.get $5
  f64x2.extract_lane 0
  local.get $5
  f64x2.extract_lane 1
  f64.add
  local.get $6
  f64x2.extract_lane 0
  f64.add
  local.get $6
  f64x2.extract_lane 1
  f64.add
  local.set $3
  loop $while-continue|1
   local.get $2
   local.get $4
   i32.gt_u
   if
    local.get $1
    local.get $4
    i32.const 3
    i32.shr_u
    i32.add
    i32.load8_u
    local.get $4
    i32.const 7
    i32.and
    i32.shr_u
    i32.const 1
    i32.and
    if
     local.get $3
     local.get $0
     local.get $4
     i32.const 3
     i32.shl
     i32.add
     f64.load
     f64.add
     local.set $3
    end
    local.get $4
    i32.const 1
    i32.add
    local.set $4
    br $while-continue|1
   end
  end
  local.get $3
 )
 (func $assembly/simd/cmp_gt_f64_mask (param $0 i32) (param $1 f64) (param $2 i32) (param $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 v128)
  (local $7 i32)
  (local $8 v128)
  local.get $3
  i32.const 7
  i32.add
  i32.const 3
  i32.shr_u
  local.set $7
  loop $for-loop|0
   local.get $5
   local.get $7
   i32.lt_u
   if
    local.get $2
    local.get $5
    i32.add
    i32.const 0
    i32.store8
    local.get $5
    i32.const 1
    i32.add
    local.set $5
    br $for-loop|0
   end
  end
  local.get $1
  f64x2.splat
  local.set $6
  loop $while-continue|1
   local.get $4
   i32.const 7
   i32.add
   local.get $3
   i32.lt_u
   if
    local.get $2
    local.get $4
    i32.const 3
    i32.shr_u
    i32.add
    local.get $0
    local.get $4
    i32.const 3
    i32.shl
    i32.add
    local.tee $5
    v128.load
    local.get $6
    f64x2.gt
    local.tee $8
    i64x2.extract_lane 0
    i64.const 1
    i64.and
    local.get $8
    i64x2.extract_lane 1
    i64.const 1
    i64.and
    i64.const 1
    i64.shl
    i64.or
    local.get $5
    v128.load offset=16
    local.get $6
    f64x2.gt
    local.tee $8
    i64x2.extract_lane 0
    i64.const 1
    i64.and
    i64.const 2
    i64.shl
    i64.or
    local.get $8
    i64x2.extract_lane 1
    i64.const 1
    i64.and
    i64.const 3
    i64.shl
    i64.or
    local.get $5
    v128.load offset=32
    local.get $6
    f64x2.gt
    local.tee $8
    i64x2.extract_lane 0
    i64.const 1
    i64.and
    i64.const 4
    i64.shl
    i64.or
    local.get $8
    i64x2.extract_lane 1
    i64.const 1
    i64.and
    i64.const 5
    i64.shl
    i64.or
    local.get $5
    v128.load offset=48
    local.get $6
    f64x2.gt
    local.tee $8
    i64x2.extract_lane 0
    i64.const 1
    i64.and
    i64.const 6
    i64.shl
    i64.or
    local.get $8
    i64x2.extract_lane 1
    i64.const 1
    i64.and
    i64.const 7
    i64.shl
    i64.or
    i64.store8
    local.get $4
    i32.const 8
    i32.add
    local.set $4
    br $while-continue|1
   end
  end
  loop $while-continue|2
   local.get $3
   local.get $4
   i32.gt_u
   if
    local.get $0
    local.get $4
    i32.const 3
    i32.shl
    i32.add
    f64.load
    local.get $1
    f64.gt
    if
     local.get $2
     local.get $4
     i32.const 3
     i32.shr_u
     i32.add
     local.tee $5
     local.get $5
     i32.load8_u
     i32.const 1
     local.get $4
     i32.const 7
     i32.and
     i32.shl
     i32.or
     i32.store8
    end
    local.get $4
    i32.const 1
    i32.add
    local.set $4
    br $while-continue|2
   end
  end
 )
)
