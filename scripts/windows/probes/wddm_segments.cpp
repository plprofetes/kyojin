// Read-only: print the WDDM video-memory segments of every adapter via D3DKMTQueryStatistics
// (CommitLimit / BytesCommitted / BytesResident per segment). Needs no elevation.
// Build (VS Developer prompt):  cl /nologo /EHsc /W3 wddm_segments.cpp /link gdi32.lib
// Layout comes from the Windows SDK header (d3dkmthk.h), not hand-written.
#define WIN32_NO_STATUS
#include <windows.h>
#undef WIN32_NO_STATUS
#include <ntstatus.h>
#include <d3dkmthk.h>
#include <cstdio>
#include <cstring>

static const double GiB = 1024.0 * 1024.0 * 1024.0;

int main()
{
    D3DKMT_ENUMADAPTERS2 en = {};
    en.NumAdapters = 0;
    if (D3DKMTEnumAdapters2(&en) != 0 || en.NumAdapters == 0) { printf("D3DKMTEnumAdapters2 failed\n"); return 1; }
    D3DKMT_ADAPTERINFO* infos = new D3DKMT_ADAPTERINFO[en.NumAdapters];
    en.pAdapters = infos;
    if (D3DKMTEnumAdapters2(&en) != 0) { printf("D3DKMTEnumAdapters2 (2nd call) failed\n"); return 1; }

    for (UINT a = 0; a < en.NumAdapters; ++a)
    {
        LUID luid = infos[a].AdapterLuid;
        D3DKMT_QUERYSTATISTICS q = {};
        q.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
        q.AdapterLuid = luid;
        NTSTATUS st = D3DKMTQueryStatistics(&q);
        if (st != 0) { printf("adapter %u luid %lx:%lx: ADAPTER query failed 0x%lx\n", a, luid.HighPart, luid.LowPart, (unsigned long)st); continue; }
        UINT nseg = q.QueryResult.AdapterInformation.NbSegments;
        printf("adapter %u luid %lx:%lx: %u segments, %u nodes, TDR detected %u\n", a, luid.HighPart, luid.LowPart,
               nseg, q.QueryResult.AdapterInformation.NodeCount, q.QueryResult.AdapterInformation.TdrDetectedCount);
        for (UINT s = 0; s < nseg; ++s)
        {
            D3DKMT_QUERYSTATISTICS qs = {};
            qs.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
            qs.AdapterLuid = luid;
            qs.QuerySegment.SegmentId = s;
            st = D3DKMTQueryStatistics(&qs);
            if (st != 0) { printf("  segment %u: query failed 0x%lx\n", s, (unsigned long)st); continue; }
            const auto& r = qs.QueryResult.SegmentInformation;
            printf("  segment %u: CommitLimit %8.2f GiB  BytesCommitted %8.2f  BytesResident %8.2f  Aperture %u\n", s,
                   (double)r.CommitLimit / GiB, (double)r.BytesCommitted / GiB, (double)r.BytesResident / GiB, (unsigned)r.Aperture);
        }
    }
    delete[] infos;
    return 0;
}
