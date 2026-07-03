/**
 * src/temporal/index.ts
 *
 * Public surface of the temporal civil-math + tz module (v2.5a, Part 1).
 * Frame integration happens in a later task; this module is standalone.
 *
 * Exports:
 *   civil.ts  — pure integer date math: daysToCivil, civilToDays, floorDivMs,
 *               msToCivil, isoWeekday, dayOfYear, isLeapYear, date32ToFields,
 *               timestampUtcToFields
 *   tz.ts     — tz-aware extraction: getTzComponents, validateIanaZone, getIcuVersion
 *   vectorize.ts — vectorized helper: extractComponents
 */

export {
  daysToCivil,
  civilToDays,
  isLeapYear,
  dayOfYear,
  isoWeekday,
  floorDivMs,
  msToCivil,
  date32ToFields,
  timestampUtcToFields,
} from './civil.js';

export type { CivilDate, CivilDateTime, CivilFields } from './civil.js';

export {
  getTzComponents,
  validateIanaZone,
  getIcuVersion,
} from './tz.js';

export { extractComponents } from './vectorize.js';
export type { DtComponent, ExtractResult } from './vectorize.js';
