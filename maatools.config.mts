import type { FullConfig } from '@nekosu/maa-tools'

const config: FullConfig = {
  cwd: import.meta.dirname,
  maaVersion: 'latest',
  interfacePath: 'assets/interface.json',
  check: {
    override: {
      // 忽略 mpe-config 带来的报错
      // ignore warning caused by mpe-config
      // 'mpe-config': 'ignore'
    }
  }
}

export default config
