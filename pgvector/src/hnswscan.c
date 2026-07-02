#include "postgres.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"

/*
 * Algorithm 5 from paper
 */
static List *
GetScanItems(IndexScanDesc scan, Datum value, int efSearch, bool keepDiscarded)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;
	pairingheap **discarded;

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	discarded = keepDiscarded || hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL;

	return HnswSearchLayer(base, q, ep, efSearch, 0, index, support, m, false, NULL, &so->v, discarded, true, &so->tuples);
}

/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItemsWithEf(IndexScanDesc scan, int batch_size)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	return HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples);
}

/*
 * Get scan value
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	return ResumeScanItemsWithEf(scan, hnsw_ef_search);
}

static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif


static int
CompareInt64(const void *a, const void *b)
{
	int64		left = *((const int64 *) a);
	int64		right = *((const int64 *) b);

	if (left < right)
		return -1;
	if (left > right)
		return 1;
	return 0;
}

static void
SquidInitFilter(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	const char *raw = squidhnsw_allowed_patterns;
	int			capacity = 0;
	char	   *copy;
	char	   *p;

	if (so->squidFilterInitialized)
		return;

	so->squidFilterInitialized = true;
	so->squidFilterEnabled = false;
	so->squidAllowedPatterns = NULL;
	so->squidAllowedPatternCount = 0;

	if (raw == NULL || raw[0] == '\0')
		return;

	for (p = (char *) raw; *p != '\0'; p++)
	{
		if (*p == ',')
			capacity++;
	}
	capacity++;
	so->squidAllowedPatterns = palloc(sizeof(int64) * capacity);

	copy = pstrdup(raw);
	p = copy;
	while (*p != '\0')
	{
		char	   *endptr;
		int64		value;

		while (*p != '\0' && (isspace((unsigned char) *p) || *p == ','))
			p++;
		if (*p == '\0')
			break;

		value = strtoll(p, &endptr, 10);
		if (endptr == p)
			break;

		so->squidAllowedPatterns[so->squidAllowedPatternCount++] = value;
		p = endptr;
	}

	if (so->squidAllowedPatternCount > 1)
		qsort(so->squidAllowedPatterns, so->squidAllowedPatternCount, sizeof(int64), CompareInt64);

	so->squidFilterEnabled = so->squidAllowedPatternCount > 0;
}

static bool
SquidPatternAllowed(HnswScanOpaque so, int64 pattern)
{
	int			left = 0;
	int			right = so->squidAllowedPatternCount - 1;

	while (left <= right)
	{
		int			mid = left + (right - left) / 2;
		int64		value = so->squidAllowedPatterns[mid];

		if (pattern == value)
			return true;
		if (pattern < value)
			right = mid - 1;
		else
			left = mid + 1;
	}

	return false;
}

static bool
SquidAuthorizedPattern(IndexScanDesc scan, int64 pattern)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	SquidInitFilter(scan);

	if (!so->squidFilterEnabled)
		return true;

	return SquidPatternAllowed(so, pattern);
}

static bool
SquidPatternAllowedCallback(void *arg, int64 pattern)
{
	HnswScanOpaque so = (HnswScanOpaque) arg;

	if (!so->squidFilterEnabled)
		return true;

	return SquidPatternAllowed(so, pattern);
}

static int
SquidClampEf(int value)
{
	int			maxEf = squidhnsw_max_ef;

	if (maxEf < HNSW_MIN_EF_SEARCH)
		maxEf = HNSW_MAX_EF_SEARCH;
	if (maxEf > HNSW_MAX_EF_SEARCH)
		maxEf = HNSW_MAX_EF_SEARCH;

	if (value < HNSW_MIN_EF_SEARCH)
		value = HNSW_MIN_EF_SEARCH;
	if (value > maxEf)
		value = maxEf;
	return value;
}

static List *
GetScanItemsAdaptive(IndexScanDesc scan, Datum value, int baseEf)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;
	HnswSquidAdaptiveContext adaptive;

	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	adaptive.baseEf = baseEf;
	adaptive.maxEf = SquidClampEf(squidhnsw_max_ef);
	adaptive.routeSelectivity = squidhnsw_route_selectivity;
	adaptive.patternAllowed = SquidPatternAllowedCallback;
	adaptive.patternAllowedArg = so;

	return HnswSearchLayerAdaptive(base, q, ep, baseEf, 0, index, support, m, false, NULL, &so->v, true, &so->tuples, &adaptive);
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	so->squidFilterInitialized = false;
	so->squidFilterEnabled = false;
	so->squidAllowedPatterns = NULL;
	so->squidAllowedPatternCount = 0;

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->tuples = 0;
	so->previousDistance = -get_float8_infinity();


	MemoryContextReset(so->tmpCtx);
	so->squidFilterInitialized = false;
	so->squidFilterEnabled = false;
	so->squidAllowedPatterns = NULL;
	so->squidAllowedPatternCount = 0;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan, value, hnsw_ef_search, false);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;

		if (list_length(so->w) == 0)
		{
			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
				break;

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				/*
				 * Locking ensures when neighbors are read, the elements they
				 * reference will not be deleted (and replaced) during the
				 * iteration.
				 *
				 * Elements loaded into memory on previous iterations may have
				 * been deleted (and replaced), so when reading neighbors, the
				 * element version must be checked.
				 */
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}


/*
 * Fetch the next authorized tuple in the SQUIDHNSW scan.
 */
bool
squidhnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;
		int			baseEf;

		pgstat_count_index_scan(scan->indexRelation);

		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan squidhnsw index without order");

		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with squidhnsw");

		value = GetScanValue(scan);
		baseEf = SquidClampEf(Max(squidhnsw_base_ef, 1));

		SquidInitFilter(scan);

		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
		so->w = GetScanItemsAdaptive(scan, value, baseEf);
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;

		if (list_length(so->w) == 0)
			break;

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);
			pfree(element);
			pfree(sc);
			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		if (!SquidAuthorizedPattern(scan, element->patternIds[element->heaptidsLength]))
			continue;

		MemoryContextSwitchTo(oldCtx);
		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
