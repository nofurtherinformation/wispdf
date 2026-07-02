/** Frame layer (P3.2) public surface: DataFrame, Series, GroupBy, join, the runtime loader
 * (init), and FrameError. Expression builders (col/lit/Expr) come from the expression layer. */

export {
  DataFrame,
  scope,
  type FrameOptions,
  type SortOptions,
  type WithColumnOptions,
} from './dataframe.js';

export { Series } from './series.js';

export {
  GroupBy,
  type AggName,
  type AggRequest,
  type AggSpec,
  type NamedColumn,
  type GroupBySource,
} from './groupby.js';

export { type JoinHow, type JoinOptions, type JoinSource } from './join.js';

export { type Row, type RowCursor } from './rowproxy.js';

export {
  init,
  loadRuntime,
  runtimeFromExports,
  useRuntime,
  defaultRuntime,
  type DfRuntime,
  type FrameWasm,
} from './runtime.js';

export { FrameError } from './errors.js';
