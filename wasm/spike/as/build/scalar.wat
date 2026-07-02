(module
 (type $0 (func (param i32) (result i32)))
 (type $1 (func (param i32 i32 i32 i32)))
 (type $2 (func (param i32 i32 i32) (result f64)))
 (type $3 (func (param i32 f64 i32 i32)))
 (global $assembly/scalar/_bump (mut i32) (i32.const 0))
 (memory $0 0)
 (export "alloc" (func $assembly/scalar/alloc))
 (export "add_f64" (func $assembly/scalar/add_f64))
 (export "sum_f64_null" (func $assembly/scalar/sum_f64_null))
 (export "cmp_gt_f64_mask" (func $assembly/scalar/cmp_gt_f64_mask))
 (export "memory" (memory $0))
 (func $assembly/scalar/alloc (param $0 i32) (result i32)
  (local $1 i32)
  global.get $assembly/scalar/_bump
  i32.eqz
  if
   i32.const 1024
   global.set $assembly/scalar/_bump
  end
  global.get $assembly/scalar/_bump
  local.set $1
  global.get $assembly/scalar/_bump
  local.get $0
  i32.const 7
  i32.add
  i32.const -8
  i32.and
  i32.add
  global.set $assembly/scalar/_bump
  global.get $assembly/scalar/_bump
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
 (func $assembly/scalar/add_f64 (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32)
  (local $4 i32)
  (local $5 i32)
  loop $for-loop|0
   local.get $3
   local.get $4
   i32.gt_u
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
    f64.load
    local.get $1
    local.get $5
    i32.add
    f64.load
    f64.add
    f64.store
    local.get $4
    i32.const 1
    i32.add
    local.set $4
    br $for-loop|0
   end
  end
 )
 (func $assembly/scalar/sum_f64_null (param $0 i32) (param $1 i32) (param $2 i32) (result f64)
  (local $3 f64)
  (local $4 f64)
  (local $5 f64)
  (local $6 f64)
  (local $7 i32)
  (local $8 i32)
  (local $9 i32)
  loop $while-continue|0
   local.get $9
   i32.const 3
   i32.add
   local.get $2
   i32.lt_u
   if
    local.get $9
    i32.const 3
    i32.shr_u
    local.set $7
    local.get $9
    i32.const 7
    i32.and
    local.tee $8
    i32.const 5
    i32.lt_u
    if (result i32)
     local.get $1
     local.get $7
     i32.add
     i32.load8_u
     local.tee $7
     local.get $8
     i32.shr_u
     i32.const 1
     i32.and
     if
      local.get $3
      local.get $0
      local.get $9
      i32.const 3
      i32.shl
      i32.add
      f64.load
      f64.add
      local.set $3
     end
     local.get $7
     local.get $8
     i32.const 1
     i32.add
     i32.shr_u
     i32.const 1
     i32.and
     if
      local.get $4
      local.get $0
      local.get $9
      i32.const 1
      i32.add
      i32.const 3
      i32.shl
      i32.add
      f64.load
      f64.add
      local.set $4
     end
     local.get $7
     local.get $8
     i32.const 2
     i32.add
     i32.shr_u
     i32.const 1
     i32.and
     if
      local.get $6
      local.get $0
      local.get $9
      i32.const 2
      i32.add
      i32.const 3
      i32.shl
      i32.add
      f64.load
      f64.add
      local.set $6
     end
     local.get $7
     local.get $8
     i32.const 3
     i32.add
     i32.shr_u
     i32.const 1
     i32.and
     if
      local.get $5
      local.get $0
      local.get $9
      i32.const 3
      i32.add
      i32.const 3
      i32.shl
      i32.add
      f64.load
      f64.add
      local.set $5
     end
     local.get $9
     i32.const 4
     i32.add
    else
     local.get $1
     local.get $7
     i32.add
     i32.load8_u
     local.get $8
     i32.shr_u
     i32.const 1
     i32.and
     if
      local.get $3
      local.get $0
      local.get $9
      i32.const 3
      i32.shl
      i32.add
      f64.load
      f64.add
      local.set $3
     end
     local.get $9
     i32.const 1
     i32.add
     local.set $8
     i32.const 0
     local.set $9
     loop $for-loop|1
      local.get $9
      i32.const 3
      i32.lt_u
      if
       local.get $1
       local.get $8
       local.get $9
       i32.add
       local.tee $7
       i32.const 3
       i32.shr_u
       i32.add
       i32.load8_u
       local.get $7
       i32.const 7
       i32.and
       i32.shr_u
       i32.const 1
       i32.and
       if
        local.get $4
        local.get $0
        local.get $7
        i32.const 3
        i32.shl
        i32.add
        f64.load
        f64.add
        local.set $4
       end
       local.get $9
       i32.const 1
       i32.add
       local.set $9
       br $for-loop|1
      end
     end
     local.get $8
     i32.const 3
     i32.add
    end
    local.set $9
    br $while-continue|0
   end
  end
  loop $while-continue|2
   local.get $2
   local.get $9
   i32.gt_u
   if
    local.get $1
    local.get $9
    i32.const 3
    i32.shr_u
    i32.add
    i32.load8_u
    local.get $9
    i32.const 7
    i32.and
    i32.shr_u
    i32.const 1
    i32.and
    if
     local.get $3
     local.get $0
     local.get $9
     i32.const 3
     i32.shl
     i32.add
     f64.load
     f64.add
     local.set $3
    end
    local.get $9
    i32.const 1
    i32.add
    local.set $9
    br $while-continue|2
   end
  end
  local.get $3
  local.get $4
  f64.add
  local.get $6
  f64.add
  local.get $5
  f64.add
 )
 (func $assembly/scalar/cmp_gt_f64_mask (param $0 i32) (param $1 f64) (param $2 i32) (param $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  local.get $3
  i32.const 7
  i32.add
  i32.const 3
  i32.shr_u
  local.set $6
  loop $for-loop|0
   local.get $4
   local.get $6
   i32.lt_u
   if
    local.get $2
    local.get $4
    i32.add
    i32.const 0
    i32.store8
    local.get $4
    i32.const 1
    i32.add
    local.set $4
    br $for-loop|0
   end
  end
  loop $while-continue|1
   local.get $5
   i32.const 7
   i32.add
   local.get $3
   i32.lt_u
   if
    local.get $2
    local.get $5
    i32.const 3
    i32.shr_u
    i32.add
    local.get $1
    local.get $0
    local.get $5
    i32.const 3
    i32.shl
    i32.add
    local.tee $4
    f64.load
    f64.lt
    local.tee $6
    i32.const 2
    i32.or
    local.get $6
    local.get $4
    f64.load offset=8
    local.get $1
    f64.gt
    select
    local.tee $6
    i32.const 4
    i32.or
    local.get $6
    local.get $4
    f64.load offset=16
    local.get $1
    f64.gt
    select
    local.tee $6
    i32.const 8
    i32.or
    local.get $6
    local.get $4
    f64.load offset=24
    local.get $1
    f64.gt
    select
    local.tee $6
    i32.const 16
    i32.or
    local.get $6
    local.get $4
    f64.load offset=32
    local.get $1
    f64.gt
    select
    local.tee $6
    i32.const 32
    i32.or
    local.get $6
    local.get $4
    f64.load offset=40
    local.get $1
    f64.gt
    select
    local.tee $6
    i32.const 64
    i32.or
    local.get $6
    local.get $4
    f64.load offset=48
    local.get $1
    f64.gt
    select
    local.tee $6
    i32.const 128
    i32.or
    local.get $6
    local.get $4
    f64.load offset=56
    local.get $1
    f64.gt
    select
    i32.store8
    local.get $5
    i32.const 8
    i32.add
    local.set $5
    br $while-continue|1
   end
  end
  loop $while-continue|2
   local.get $3
   local.get $5
   i32.gt_u
   if
    local.get $0
    local.get $5
    i32.const 3
    i32.shl
    i32.add
    f64.load
    local.get $1
    f64.gt
    if
     local.get $2
     local.get $5
     i32.const 3
     i32.shr_u
     i32.add
     local.tee $4
     local.get $4
     i32.load8_u
     i32.const 1
     local.get $5
     i32.const 7
     i32.and
     i32.shl
     i32.or
     i32.store8
    end
    local.get $5
    i32.const 1
    i32.add
    local.set $5
    br $while-continue|2
   end
  end
 )
)
