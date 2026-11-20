<#
.SYNOPSIS
  Print the WDDM video-memory budget per adapter (DXGI IDXGIAdapter3::QueryVideoMemoryInfo): Budget, CurrentUsage
  (of THIS process, so ~0 here), AvailableForReservation, for the local and the non-local (shared) segment group.
  Run it while another process holds GPU memory to see how the budget reacts. Read-only.
#>
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Dxgi {
  [StructLayout(LayoutKind.Sequential)] public struct MemInfo { public ulong Budget, CurrentUsage, AvailableForReservation, CurrentReservation; }
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] public struct AdapterDesc1 {
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Description;
    public uint VendorId, DeviceId, SubSysId, Revision; public UIntPtr DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
    public long Luid; public uint Flags; }
  [ComImport, Guid("7b7166ec-21c7-44ae-b21a-c9ae321ae369"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IFactory { void a(); void b(); void c(); void d(); void e(); void f(); // IDXGIObject(4) + EnumAdapters, MakeWindowAssociation, GetWindowAssociation
    [PreserveSig] int g(); [PreserveSig] int h(); }
  [DllImport("dxgi.dll")] static extern int CreateDXGIFactory1(ref Guid riid, out IntPtr ppFactory);
  delegate int EnumAdapters1Fn(IntPtr self, uint i, out IntPtr adapter);
  delegate int GetDesc1Fn(IntPtr self, out AdapterDesc1 d);
  delegate int QueryFn(IntPtr self, uint node, int group, out MemInfo info);
  delegate uint ReleaseFn(IntPtr self);
  static IntPtr Slot(IntPtr obj, int index) { IntPtr vt = Marshal.ReadIntPtr(obj); return Marshal.ReadIntPtr(vt, index * IntPtr.Size); }
  public static void Run() {
    Guid fid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387"); // IDXGIFactory1
    IntPtr fac; int hr = CreateDXGIFactory1(ref fid, out fac); if (hr != 0) { Console.WriteLine("CreateDXGIFactory1 failed 0x" + hr.ToString("x")); return; }
    var enumFn = (EnumAdapters1Fn)Marshal.GetDelegateForFunctionPointer(Slot(fac, 12), typeof(EnumAdapters1Fn)); // IDXGIFactory1::EnumAdapters1
    for (uint i = 0; ; i++) {
      IntPtr ad; if (enumFn(fac, i, out ad) != 0) break;
      var desc = (GetDesc1Fn)Marshal.GetDelegateForFunctionPointer(Slot(ad, 10), typeof(GetDesc1Fn)); // IDXGIAdapter1::GetDesc1
      AdapterDesc1 d; desc(ad, out d);
      Console.WriteLine("adapter {0}: {1}  dedicated {2:N2} GiB  shared-system {3:N2} GiB", i, d.Description, (double)(ulong)d.DedicatedVideoMemory / 1073741824.0, (double)(ulong)d.SharedSystemMemory / 1073741824.0);
      Guid a3 = new Guid("645967a4-1392-4310-a798-8053ce3e93fd"); IntPtr p3;
      if (Marshal.QueryInterface(ad, ref a3, out p3) == 0) {
        var q = (QueryFn)Marshal.GetDelegateForFunctionPointer(Slot(p3, 14), typeof(QueryFn)); // IDXGIAdapter3::QueryVideoMemoryInfo
        for (int g = 0; g < 2; g++) { MemInfo m; int r = q(p3, 0, g, out m);
          Console.WriteLine("  {0,-9} hr=0x{1:x} budget {2,8:N2} GiB  usage(this process) {3,6:N2}  availForReservation {4,8:N2}  reservation {5:N2}",
            g == 0 ? "local" : "non-local", r, m.Budget / 1073741824.0, m.CurrentUsage / 1073741824.0, m.AvailableForReservation / 1073741824.0, m.CurrentReservation / 1073741824.0); }
        ((ReleaseFn)Marshal.GetDelegateForFunctionPointer(Slot(p3, 2), typeof(ReleaseFn)))(p3);
      }
      ((ReleaseFn)Marshal.GetDelegateForFunctionPointer(Slot(ad, 2), typeof(ReleaseFn)))(ad);
    }
  }
}
'@
[Dxgi]::Run()
